import os

def format_diff_to_prompt(diff_file):
    with open(diff_file, 'r') as f:
        lines = f.readlines()
    
    def find_section_start(keyword):
        return next((i for i, line in enumerate(lines) if line.startswith(keyword)), None)
    
    commit_idx = find_section_start("Commit:")
    author_idx = find_section_start("Author:")
    date_idx = find_section_start("Date:")
    message_idx = find_section_start("Full Message:")
    diff_idx = find_section_start("Diff:")
    
    if any(idx is None for idx in [commit_idx, author_idx, date_idx, message_idx, diff_idx]):
        raise ValueError("Missing section in diff file.")
    
    commit_value = lines[commit_idx].split(": ", 1)[1].strip()
    author_value = lines[author_idx].split(": ", 1)[1].strip()
    date_value = lines[date_idx].split(": ", 1)[1].strip()
    message = "".join(lines[message_idx+1:diff_idx]).strip()
    diff = "".join(lines[diff_idx+1:]).strip()
    
    prompt = "Based on this commit message and diff, summarize the purpose and impact in a paragraph:\n\n"
    formatted = f"{prompt}On {date_value}, {author_value} created commit {commit_value}, and stated:\n\n{message}\n\nDiff:\n{diff}"
    
    # Optional: Save modified (to PVC via caller)
    modified_file = os.path.join(os.path.dirname(diff_file), f"modified_{os.path.basename(diff_file)}")
    with open(modified_file, 'w') as f:
        f.write(formatted)
    
    return formatted

# For testing
if __name__ == "__main__":
    diff_dir = os.path.join(os.environ.get('WORKSPACE_ROOT', '/opt/app-root/src/workspace'), 'diffs/assisted-service')
    for file in os.listdir(diff_dir):
        if file.endswith("_diff.txt"):
            format_diff_to_prompt(os.path.join(diff_dir, file))