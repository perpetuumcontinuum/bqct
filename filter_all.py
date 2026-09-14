import json
import os
import re
import time
from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-3b-instruct-q5_k_m.gguf"
CORE_FILE = "core.json"
PROGRESS_FILE = "progress.json"

BATCH_SIZE = 20
N_THREADS = os.cpu_count() or 4
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "19000"))
START_TIME = time.time()

print(f"Threads: {N_THREADS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max runtime: {MAX_RUNTIME_SEC}s")

print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=N_THREADS,
    n_threads_batch=N_THREADS,
    flash_attn=True,
    verbose=False,
)

SYSTEM = "You classify English words. Answer briefly and exactly."

def classify_batch(batch):
    prompt = (
        "Classify each word below as KEEP or REMOVE.\n\n"
        "KEEP if it is a real common English word (noun, verb, adjective).\n"
        "REMOVE if it is a person's name, brand, or nonsense word.\n\n"
        "Examples:\n"
        "love KEEP\n"
        "man KEEP\n"
        "john REMOVE\n"
        "kardashian REMOVE\n"
        "rose KEEP\n"
        "googoo REMOVE\n"
        "smith KEEP\n\n"
        "Answer one word per line: word KEEP or word REMOVE.\n\n"
        "Words: " + ", ".join(batch)
    )
    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        text = result["choices"][0]["message"]["content"]
        print(f"\n=== QWEN RESPONSE (first 800 chars) ===\n{text[:800]}\n=== END ===\n")
        return text
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""

def parse_response(text):
    remove = []
    for line in text.splitlines():
        line = line.strip().lower()
        if not line:
            continue
        m = re.match(r'^([a-z]+)[\s:\-\.]+(keep|remove)', line)
        if m:
            word, verdict = m.group(1), m.group(2)
            if verdict == "remove":
                remove.append(word)
    return remove

with open(CORE_FILE, "r", encoding="utf-8") as f:
    core = json.load(f)

words = [w for w in core.keys() if not w.startswith("__")][:200]
total = len(words)
print(f"Words to classify: {total}")

to_remove = set()

for i in range(0, total, BATCH_SIZE):
    batch = words[i:i+BATCH_SIZE]
    response = classify_batch(batch)
    removed = parse_response(response)
    if removed:
        print(f"  Removed: {removed}")
    to_remove.update([w.lower() for w in removed])
    print(f"[{min(i+BATCH_SIZE, total)}/{total}] removed so far: {len(to_remove)}")

print(f"\nTotal to remove: {len(to_remove)}")
print(f"To remove list: {sorted(to_remove)}")
