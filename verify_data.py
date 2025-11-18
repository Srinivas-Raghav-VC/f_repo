
import sys
import os
import re
from mmie import read_jsonl

def check_script(text, script_name):
    if script_name == "Devanagari":
        return bool(re.search(r'[\u0900-\u097F]', text))
    elif script_name == "Latin":
        return bool(re.search(r'[a-zA-Z]', text))
    return False

def verify_file(path, expected_script):
    print(f"--- Verifying {os.path.basename(path)} ---")
    try:
        data = read_jsonl(path, lim=5)
        if not data:
            print(f"[WARN] File is empty or could not be read: {path}")
            return
        
        print(f"Read {len(data)} samples.")
        for i, text in enumerate(data[:2]):
            print(f"Sample {i}: {text[:100]}...")
            if expected_script:
                has_script = check_script(text, expected_script)
                print(f"  Contains {expected_script}: {has_script}")
                if not has_script:
                    print(f"  [WARN] Expected {expected_script} but not found in sample!")
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")

def main():
    base_dir = "data"
    verify_file(os.path.join(base_dir, "forget_hi.jsonl"), "Devanagari")
    verify_file(os.path.join(base_dir, "retain_en.jsonl"), "Latin")
    verify_file(os.path.join(base_dir, "mixed.jsonl"), None) # Mixed might have both

if __name__ == "__main__":
    main()
