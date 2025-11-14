import os
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import time
import re

# Env workspace_root
workspace_root = os.environ.get('WORKSPACE_ROOT', '/opt/app-root/src/workspace')

# Logging to PVC
os.makedirs(os.path.join(workspace_root, 'outputs'), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_config(config_file=None, config_dict=None):
    if config_dict:
        return config_dict
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Failed to parse config file {config_file}: {e}")
        raise
    return config

def extract_section(lines, startswith):
    for line in lines:
        if line.startswith(startswith):
            return line.split(": ", 1)[1].strip()
    return None

def clean_content(content):
    cleaned = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith('#') or line.startswith('Signed-off-by:') or not line or line.startswith('@@') or line.startswith('diff --git'):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def parse_diff_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        commit = extract_section(lines, "Commit:")
        author = extract_section(lines, "Author:")
        date = extract_section(lines, "Date:")
        message_start = next((i for i, line in enumerate(lines) if line.startswith("Full Message:")), None)
        before_start = next((i for i, line in enumerate(lines) if line.startswith("Before change:")), None)
        after_start = next((i for i, line in enumerate(lines) if line.startswith("After change:")), None)
        diff_start = next((i for i, line in enumerate(lines) if line.startswith("Diff:")), None)
        
        if any(x is None for x in [commit, author, date, message_start, diff_start]):
            logger.error(f"Missing required section in {file_path}")
            return None
        
        raw_message = "".join(lines[message_start+1:(before_start or after_start or diff_start)]).strip()
        raw_before = "".join(lines[before_start+1:(after_start or diff_start)]).strip() if before_start else ""
        raw_after = "".join(lines[after_start+1:diff_start]).strip() if after_start else ""
        raw_diff = "".join(lines[diff_start+1:]).strip()[:5000]
        
        # Clean before returning
        message = clean_content(raw_message)
        before = clean_content(raw_before)
        after = clean_content(raw_after)
        diff = clean_content(raw_diff)
        return commit, author, date, message, before, after, diff
    except Exception as e:
        logger.error(f"Failed to parse diff file {file_path}: {e}")
        return None

def generate_summary_from_diff(diff_file, model, tokenizer, file_context="", temperature=0.7, max_tokens=2048):
    start_time = time.time()
    result = parse_diff_file(diff_file)
    if result is None:
        return {"summary": "Summary skipped due to parsing error.", "user_story": ""}
    commit, author, date, message, before, after, diff = result
    
    # Few-shot examples for summary (simple diff + ideal output, including before/after)
    few_shot = (
        "Example 1: Before change:\n  enum:\n    - 'old-feature'\nAfter change:\n  enum:\n    - 'old-feature'\n    - 'new-feature'\nDiff:\n+ - 'new-feature'\nSummary: This change adds a new feature option to the API schema, enabling support for additional functionality in cluster deployments.\n\n"
        "Example 2: Before change:\n  validation:\n    - 'item1'\nAfter change:\n  validation:\n    - 'item1'\n    - 'item2'\n    - 'item3'\nDiff:\n+ - 'item2'\n+ - 'item3'\nSummary: This change expands the validation list in the CRD, adding two new items to enhance security checks during bare metal provisioning.\n\n"
    )
    
    # Chained prompt: Summary first
    prompt_summary = f"{file_context}\n\n{few_shot}Based on this commit message, before/after content, and diff, generate a concise summary of the purpose and impact in 1-2 sentences:\n\nOn {date}, {author} created commit {commit}, and stated:\n\n{message}\n\nBefore change:\n{before}\n\nAfter change:\n{after}\n\nDiff:\n{diff}\n\nSummary:"
    inputs_summary = tokenizer.encode(prompt_summary, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs_summary = model.generate(
            inputs_summary, max_new_tokens=max_tokens//2, temperature=temperature, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    summary = tokenizer.decode(outputs_summary[0][len(inputs_summary[0]):], skip_special_tokens=True).strip()
    
    # User story chain
    prompt_user_story = f"{summary}\n\nBased on the summary above, generate a user story in the format 'As a [role], I want [feature] so that [benefit]':\n\nUser Story:"
    inputs_us = tokenizer.encode(prompt_user_story, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs_us = model.generate(
            inputs_us, max_new_tokens=max_tokens//2, temperature=temperature, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    user_story = tokenizer.decode(outputs_us[0][len(inputs_us[0]):], skip_special_tokens=True).strip()
    
    # Robust cleanup
    summary = summary.replace('#', '').strip()
    if summary.lower().startswith('summary:') or summary.lower().startswith('ummary:'):
        summary = summary[8:].strip() if summary.lower().startswith('summary:') else summary[7:].strip()
    summary = re.sub(r"(\w)'(\w{3,})", r"\1 '\2", summary)
    user_story = user_story.strip()
    if user_story.lower().startswith('user story:'):
        user_story = user_story[11:].strip()
    user_story = re.sub(r"(\w)'(\w{3,})", r"\1 '\2", user_story)
    user_story_lines = user_story.split('\n')
    seen = set()
    cleaned_user_story = []
    for line in user_story_lines:
        if line.strip() and 'As a' in line and line not in seen:
            cleaned_user_story.append(line)
            seen.add(line)
    user_story = cleaned_user_story[0] if cleaned_user_story else user_story
    
    end_time = time.time()
    logger.info(f"Generation time: {end_time - start_time:.2f} seconds")
    
    return {
        "summary": summary,
        "user_story": user_story
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries from diff files based on config.")
    parser.add_argument("config_file", help="Path to the config file (e.g., assisted-service.conf)")
    parser.add_argument("--model-name", default=None, help="Model name to use (overrides config; default from config or 'ibm-granite/granite-8b-code-instruct')")
    args = parser.parse_args()
    
    config = parse_config(args.config_file)
    model_name = args.model_name or config.get('model_name', "ibm-granite/granite-8b-code-instruct")
    file_context = config.get('file_context', "")  # Use per-file context from config
    if file_context:
        file_context += " "
    
    # Load model dynamically after parsing
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
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
    
    project_name = os.path.basename(args.config_file).replace('.conf', '')
    diff_dir_base = config.get('diff_dir_base', 'diffs')
    summary_dir_base = config.get('summary_dir_base', 'summaries')
    
    # Set directories dynamically to PVC
    diff_dir = os.path.join(workspace_root, diff_dir_base, project_name)
    summary_dir = os.path.join(workspace_root, summary_dir_base, project_name)
    
    os.makedirs(summary_dir, exist_ok=True)
    for diff_file in os.listdir(diff_dir):
        if diff_file.endswith("_diff.txt"):
            file_path = os.path.join(diff_dir, diff_file)
            logger.info(f"Processing {diff_file}")
            enhanced_summary = generate_summary_from_diff(file_path, model, tokenizer, file_context)
            commit = diff_file.split('_')[0]
            lines = open(file_path).readlines()
            jira = extract_section(lines, 'Jira Ticket:') or 'None'
            if jira.endswith(':'):
                jira = jira[:-1]
            author = extract_section(lines, 'Author:') or 'Unknown'
            date = extract_section(lines, 'Date:') or 'Unknown'
            summary_file = os.path.join(summary_dir, f"{commit}_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "commit": commit,
                    "author": author,
                    "date": date,
                    "jira_ticket": jira,
                    **enhanced_summary
                }, f, indent=2)
            logger.info(f"Generated summary for {commit}")