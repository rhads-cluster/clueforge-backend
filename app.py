import sys
import os
# Add scripts directory to sys.path (POC)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import yaml
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
from track_changes import generate_diffs_and_summaries
from translate_diffs import format_diff_to_prompt
from generate_summaries import generate_summary_from_diff
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Env-driven paths (from Deployment envs)
project_dir = os.environ.get('PROJECT_DIR', '/opt/app-root/src')
workspace_root = os.environ.get('WORKSPACE_ROOT', f'{project_dir}/workspace')
app_config_path = os.environ.get('APP_CONFIG_PATH', '/app/config/app-config.yaml')
model_config_path = os.environ.get('MODEL_CONFIG_PATH', '/app/model/model-config.yaml')

# Ensure PVC dirs for outputs/diffs/summaries/data (idempotent)
for subdir in ['outputs', 'diffs', 'summaries', 'data']:
    os.makedirs(os.path.join(workspace_root, subdir), exist_ok=True)

# Logging to PVC outputs/app_errors.log
os.makedirs(os.path.join(workspace_root, 'outputs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(workspace_root, 'outputs', 'app_errors.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load app config from ConfigMap mount
def load_yaml_config():
    if not os.path.exists(app_config_path):
        logger.error(f"Configuration file not found at {app_config_path}. Please create a valid app-config.yaml.")
        raise FileNotFoundError(f"Configuration file not found at {app_config_path}. Check ConfigMap mount.")
    try:
        with open(app_config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or not isinstance(config, dict) or 'products' not in config:
            logger.error(f"Invalid app-config.yaml format at {app_config_path}. Must contain 'products' key.")
            raise ValueError(f"Invalid app-config.yaml format at {app_config_path}. Must contain 'products' key.")
        logger.info("App config loaded successfully from ConfigMap.")
        return config
    except Exception as e:
        logger.error(f"Failed to load app-config.yaml: {e}")
        raise

# Load model config from ConfigMap mount
def load_model_config():
    if not os.path.exists(model_config_path):
        logger.error(f"Model config not found at {model_config_path}. Check ConfigMap mount.")
        raise FileNotFoundError(f"Model config not found at {model_config_path}.")
    try:
        with open(model_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Model config loaded successfully from ConfigMap.")
        return config
    except Exception as e:
        logger.error(f"Failed to load model-config.yaml: {e}")
        raise

# Load configs and model at startup
app_config = load_yaml_config()
model_config = load_model_config()
model_name = model_config.get('model_name', 'ibm-granite/granite-3.3-8b-instruct')
quantization = model_config.get('quantization', '8bit')  # Map q4_0 to 8bit for BitsAndBytes
cpu_fallback = model_config.get('cpu_fallback', True)
max_tokens = model_config.get('max_tokens', 2048)
temperature = model_config.get('temperature', 0.7)
device = model_config.get('device', 'auto')

try:
    # Dynamic quant config
    if quantization in ['q4_0', '8bit']:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
    else:
        quant_config = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        device_map=device,
        torch_dtype=torch.float16
    )
    if cpu_fallback and not torch.cuda.is_available():
        model = model.to('cpu')
    model.eval()
    logger.info(f"Model '{model_name}' loaded on {model.device} (quant: {quantization})")
except Exception as e:
    logger.warning(f"GPU model load failed: {e}. Falling back to CPU.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    ).to('cpu')
    model.eval()

# Health endpoint for liveness/readiness
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_device': str(model.device) if model else 'unloaded'}), 200

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        products_data = app_config.get('products', {})
        products = [{'value': key, 'label': products_data[key].get('name', key)} for key in products_data]
        return jsonify({'products': products})
    except Exception as e:
        logger.error(f"Failed to load products: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/repos', methods=['GET'])
def get_repos():
    product = request.args.get('product')
    try:
        repos_data = app_config.get('products', {}).get(product, {}).get('repos', {})
        repos = [{'value': key, 'label': key} for key in repos_data]  # Repo keys are display-friendly already (no spaces)
        return jsonify({'repos': repos})
    except Exception as e:
        logger.error(f"Failed to load repos for product {product}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files', methods=['GET'])
def get_files():
    product = request.args.get('product')
    repo = request.args.get('repo')
    try:
        files_data = app_config.get('products', {}).get(product, {}).get('repos', {}).get(repo, {}).get('files', [])
        files = [{'value': f['name'], 'label': f['name']} for f in files_data]  # Use friendly 'name' for both (map to filename later)
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Failed to load files for {product}/{repo}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_diffs', methods=['POST'])
def gen_diffs():
    data = request.get_json()
    product = data.get('product')
    repo = data.get('repo')
    selected_file = data.get('file')  # This is now the friendly 'name'
    last_n = data.get('last_n', 0)
    since = data.get('since')
    until = data.get('until')
    logical_op = data.get('logical_op', 'AND')
    full_context = data.get('full_context', True)
    force_regenerate = data.get('force_regenerate', False)  # New: from UI checkbox

    def generate_stream():
        try:
            repo_config = app_config.get('products', {}).get(product, {}).get('repos', {}).get(repo, {})
            if not repo_config:
                yield json.dumps({'type': 'error', 'message': f"Invalid product/repo: {product}/{repo}"}) + '\n'
                return
            files_data = repo_config.get('files', [])
            file_entry = next((f for f in files_data if f['name'] == selected_file), None)
            if not file_entry:
                yield json.dumps({'type': 'error', 'message': f"Invalid file: {selected_file}"}) + '\n'
                return
            target_file = file_entry['filename']  # Map friendly to actual filename
            file_context = file_entry['file_context']

            # Create temporary config file with project_name as repo key (PVC paths)
            temp_config = {
                'project_name': repo,  # Use repo key for diffs/<repo>, etc.
                'repo_path': os.path.join(workspace_root, repo),  # Pre-cloned in PVC
                'remote_branch': repo_config.get('remote_branch', 'origin/master'),
                'data_dir': os.path.join(workspace_root, 'data'),
                'diff_dir_base': os.path.join(workspace_root, 'diffs'),
                'summary_dir_base': os.path.join(workspace_root, 'summaries'),
                'model_name': model_name,
                'file_context': file_context,
                'temperature': temperature,  # From model ConfigMap
                'max_tokens': max_tokens
            }
            temp_config_path = os.path.join(project_dir, 'configs', 'temp.conf')
            with open(temp_config_path, 'w') as f:
                for key, value in temp_config.items():
                    f.write(f"{key}={value}\n")

            # Stream results from generator (pass model/tokenizer/temperature/max_tokens)
            for chunk in generate_diffs_and_summaries(
                temp_config_path,
                target_file,
                last_n=bool(last_n),
                last_n_count=last_n or 10,
                since=since,
                until=until,
                logical_op=logical_op,
                full_context=full_context,
                model=model,
                tokenizer=tokenizer,
                force_regenerate=force_regenerate,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield json.dumps(chunk) + '\n'
            
            os.remove(temp_config_path)
        except Exception as e:
            logger.error(f"Failed to generate diffs: {e}")
            yield json.dumps({'type': 'error', 'message': str(e)}) + '\n'

    return Response(stream_with_context(generate_stream()), mimetype='application/x-ndjson')

@app.route('/api/format_prompt', methods=['GET'])
def format_prompt():
    diff_file = request.args.get('diff_file')
    try:
        prompt = format_diff_to_prompt(diff_file)
        return jsonify({'prompt': prompt})
    except Exception as e:
        logger.error(f"Failed to format prompt for {diff_file}: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/generate_summary', methods=['GET'])
def gen_summary():
    diff_file = request.args.get('diff_file')
    try:
        # Use default file_context if not specific to file
        summary = generate_summary_from_diff(
            diff_file, model, tokenizer, app_config.get('file_context', ''), 
            temperature=temperature, max_tokens=max_tokens
        )
        return jsonify({'summary': summary})
    except Exception as e:
        logger.error(f"Failed to generate summary for {diff_file}: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Production: debug=False