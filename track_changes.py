import os
import json
import logging
from git import Repo
import argparse
from datetime import datetime
from generate_summaries import generate_summary_from_diff
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
import yaml
import re

# Env workspace_root (PVC)
workspace_root = os.environ.get('WORKSPACE_ROOT', '/opt/app-root/src/workspace')

# Logging to PVC
os.makedirs(os.path.join(workspace_root, 'outputs'), exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(workspace_root, 'outputs', 'app_errors.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

def parse_config(config_file=None, config_dict=None):
    if config_dict:
        return config_dict  # YAML-loaded dict from app.py temp.conf
    config = {}
    try:
        logger.debug(f"Parsing config file: {config_file}")
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        logger.debug(f"Config parsed: {config}")
    except Exception as e:
        logger.error(f"Failed to parse config file {config_file}: {e}")
        raise
    return config

def clean_context(content):
    """Clean context by removing git metadata, comments, and non-content lines."""
    cleaned = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('Signed-off-by:') or stripped.startswith('diff --git') or stripped.startswith('@@') or not stripped:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def load_install_config_schema(repo_path, target_file):
    """Load schema for install-config.yaml or other CRDs to extract field properties."""
    schema_path = None
    if 'install-config.yaml' in target_file or 'install.openshift.io_installconfigs.yaml' in target_file:
        schema_path = os.path.join(repo_path, 'data/data/install.openshift.io_installconfigs.yaml')
    elif 'metal3.io' in target_file:
        schema_path = target_file  # Use the CRD file itself as it contains schema
    try:
        if schema_path and os.path.exists(schema_path):
            logger.debug(f"Loading schema: {schema_path}")
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
            properties = schema.get('spec', {}).get('versions', [{}])[0].get('schema', {}).get('openAPIV3Schema', {}).get('properties', {})
            logger.debug(f"Schema properties: {list(properties.keys())}")
            return properties
        else:
            logger.debug(f"No schema found for {target_file}")
            return {}
    except Exception as e:
        logger.error(f"Failed to load schema {schema_path}: {e}")
        return {}

def generate_diffs_and_summaries(config_file=None, config_dict=None, target_file=None, last_n=False, last_n_count=10, since=None, until=None, logical_op='AND', full_context=True, model=None, tokenizer=None, force_regenerate=False, temperature=0.7, max_tokens=2048):
    if config_dict is None and config_file is None:
        logger.error("Config required (file or dict).")
        yield {'type': 'error', 'message': "Config required."}
        return
    
    if config_file:
        config = parse_config(config_file)
    else:
        config = parse_config(config_dict=config_dict)
    
    repo_path = os.path.expanduser(config.get('repo_path'))
    remote_branch = config.get('remote_branch')
    project_name = config.get('project_name', os.path.basename(repo_path))
    file_context = config.get('file_context', '')
    
    data_dir = os.path.join(workspace_root, config.get('data_dir', 'data'))
    diff_dir_base = os.path.join(workspace_root, config.get('diff_dir_base', 'diffs'))
    summary_dir_base = os.path.join(workspace_root, config.get('summary_dir_base', 'summaries'))
    
    if not all([repo_path, remote_branch]):
        logger.error("Config must include repo_path, remote_branch")
        yield {'type': 'error', 'message': "Config must include repo_path, remote_branch"}
        return
    
    logger.debug(f"Config: repo_path={repo_path}, target_file={target_file}, remote_branch={remote_branch}, project_name={project_name}, file_context={file_context}")
    
    last_commit_file = os.path.join(data_dir, f'{project_name}_last_processed_commit.txt')
    diff_dir = os.path.join(diff_dir_base, project_name)
    summary_dir = os.path.join(summary_dir_base, project_name)
    
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    try:
        repo = Repo(repo_path)
        current_commit = repo.head.commit.hexsha
        logger.debug(f"Current commit: {current_commit}")
        
        if last_n:
            commits = list(repo.iter_commits(remote_branch, max_count=last_n_count))[:last_n_count]
        else:
            commits = list(repo.iter_commits(remote_branch, since=since, until=until))
            if logical_op == 'OR':
                # Pseudo-OR logic: Union of since/until if both set
                if since and until:
                    commits = list(repo.iter_commits(remote_branch, since=since)) + list(repo.iter_commits(remote_branch, until=until))
                    commits = list(set(commits))  # Dedupe
            logger.debug(f"Commits to process: {len(commits)}")
        
        last_processed = None
        if os.path.exists(last_commit_file):
            with open(last_commit_file, 'r') as f:
                last_processed = f.read().strip()
        
        for commit in commits:
            commit_hash = commit.hexsha
            if last_processed and not last_n and commit_hash == last_processed:
                logger.debug(f"Commit {commit_hash} already processed; skipping")
                continue
            
            try:
                author = commit.author.name
                date = commit.authored_datetime.strftime('%Y-%m-%d %H:%M:%S')
                message = commit.message.strip()
                jira_ticket = re.search(r'(MGMT-\d+|OCPBUGS-\d+)', message) or 'None'
                jira_ticket = jira_ticket.group() if jira_ticket else 'None'
                
                # Diff for target_file
                diff = commit.diff(commit.parents[0] if commit.parents else None, paths=target_file) if commit.parents else []
                changes = []
                for d in diff:
                    if d.change_type == 'M':
                        changes.append({
                            'type': d.change_type,
                            'path': d.a_path,
                            'lines_added': d.diff.decode().count('+'),
                            'lines_deleted': d.diff.decode().count('-')
                        })
                
                if not changes and not force_regenerate:
                    logger.debug(f"No changes in {target_file} for {commit_hash}; skipping")
                    continue
                
                # Full diff text with before/after if full_context
                if full_context:
                    try:
                        before_content = repo.git.show(f"{commit.parents[0].hexsha}:{target_file}") if commit.parents else ""
                        after_content = repo.git.show(f"{commit_hash}:{target_file}")
                        diff_text = repo.git.diff(f"{commit.parents[0].hexsha}:{target_file}", f"{commit_hash}:{target_file}")
                    except Exception as e:
                        logger.warning(f"Full context failed for {commit_hash}: {e}")
                        diff_text = repo.git.diff(commit.parents[0].hexsha, commit_hash, name_only=False) if commit.parents else ""
                        before_content = after_content = ""
                else:
                    diff_text = repo.git.diff(commit.parents[0].hexsha, commit_hash, name_only=False) if commit.parents else ""
                    before_content = after_content = ""
                
                # Clean diff
                diff_text = clean_context(diff_text)
                
                # Write diff file to PVC
                diff_file = os.path.join(diff_dir, f"{commit_hash}_diff.txt")
                with open(diff_file, 'w') as f:
                    f.write(f"Commit: {commit_hash}\n")
                    f.write(f"Author: {author}\n")
                    f.write(f"Date: {date}\n")
                    f.write(f"Full Message: {message}\n")
                    if full_context:
                        f.write("Before change:\n" + before_content + "\n")
                        f.write("After change:\n" + after_content + "\n")
                    f.write("Diff:\n" + diff_text + "\n")
                    f.write(f"Jira Ticket: {jira_ticket}\n")
                
                # Generate summary (pass temperature/max_tokens)
                summary_text = None
                if model and tokenizer:
                    try:
                        logger.debug(f"Generating summary for {diff_file} with file_context: {file_context}")
                        summary_text = generate_summary_from_diff(
                            diff_file, model, tokenizer, file_context,
                            temperature=temperature, max_tokens=max_tokens
                        )
                        logger.debug(f"Summary generated: {summary_text}")
                    except Exception as e:
                        logger.warning(f"AI summary failed for {commit_hash}: {e}")
                        yield {'type': 'error', 'message': str(e), 'commit': commit_hash}
                        continue
                
                full_summary = {
                    "commit": commit_hash,
                    "author": author,
                    "date": date,
                    "jira_ticket": jira_ticket,
                    "parsed_changes": changes,
                    **(summary_text if isinstance(summary_text, dict) else {"summary": summary_text})
                }
                
                # Write summary to PVC
                summary_file = os.path.join(summary_dir, f"{commit_hash}_summary.json")
                with open(summary_file, 'w') as f:
                    json.dump(full_summary, f, indent=2)
                logger.debug(f"Summary file written: {summary_file}")
                
                yield {'type': 'result', 'data': {
                    "commit": commit_hash,
                    "author": author,
                    "date": date,
                    "jira_ticket": jira_ticket,
                    "summary_file": summary_file,
                    "summary": summary_text.get("summary", "") if isinstance(summary_text, dict) else summary_text,
                    "user_story": summary_text.get("user_story", "") if isinstance(summary_text, dict) else ""
                }}
            except Exception as e:
                logger.error(f"Error processing commit {commit_hash}: {e}")
                yield {'type': 'error', 'message': str(e), 'commit': commit_hash}
        
        if not last_n:
            try:
                with open(last_commit_file, 'w') as f:
                    f.write(current_commit)
                logger.debug(f"Last commit file written: {last_commit_file}")
            except Exception as e:
                logger.error(f"Failed to write last commit file {last_commit_file}: {e}")
    except Exception as e:
        logger.error(f"Repo ops failed: {e}")
        yield {'type': 'error', 'message': str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--target-file", required=True, help="Target file to process (e.g., install-config.yaml)")
    parser.add_argument("--last-n", type=int, default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--logical-op", default='AND')
    parser.add_argument("--model-name", default=None, help="Model name to use (overrides config; default from config or 'ibm-granite/granite-8b-code-instruct')")
    parser.add_argument("--no-full-context", action='store_false', dest='full_context', help="Disable including excerpts from full old/new file in diff_file")
    parser.add_argument("--force-regenerate", action='store_true', help="Force regeneration of summaries even if they exist")
    args = parser.parse_args()
    
    config = parse_config(args.config_file)
    model_name = args.model_name or config.get('model_name', "ibm-granite/granite-8b-code-instruct")
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
    try:
        logger.debug(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    except Exception as e:
        logger.warning(f"GPU load failed: {e}. Falling back to CPU.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        ).to('cpu')
    model.eval()
    
    try:
        for chunk in generate_diffs_and_summaries(
            args.config_file,
            target_file=args.target_file,
            last_n=bool(args.last_n),
            last_n_count=args.last_n or 10,
            since=args.since,
            until=args.until,
            logical_op=args.logical_op,
            full_context=args.full_context,
            model=model,
            tokenizer=tokenizer,
            force_regenerate=args.force_regenerate,
            temperature=0.7,  # Default; override via config
            max_tokens=2048
        ):
            print(json.dumps(chunk))
    except Exception as e:
        logger.error(f"Error in generate_diffs_and_summaries: {e}")