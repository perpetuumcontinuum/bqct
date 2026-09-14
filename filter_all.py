import json
import os
import re
import time
from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-3b-instruct-q5_k_m.gguf"
CORE_FILE = "core.json"
PROGRESS_FILE = "progress.json"

BATCH_SIZE = 200
N_THREADS = os.cpu_count() or 4
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "19000"))
START_TIME = time.time()

print(f"Threads: {N_THREADS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max runtime: {MAX_RUNTIME_SEC}s")

# Load progress
RESUME_FROM = 0
to_remove = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        prog = json.load(f)
        to_remove = set(prog.get("to_remove", []))
        RESUME_FROM = prog.get("processed", 0)
        print(f"Resumed: processed={RESUME_FROM}, to_remove={len(to_remove)}")

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
        "For each word below, decide:\n"
        "- KEEP if it is a real common English word (noun, verb, adjective).\n"
        "- REMOVE if it is a name, brand, place, or a nonsense word.\n\n"
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
        return text
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""

def parse_response(text):
    """Parse lines like 'word KEEP' or 'word REMOVE'."""
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

words = [w for w in core.keys() if not w.startswith("__")]
total = len(words)
print(f"Words to classify: {total}")

i = RESUME_FROM
while i < total:
    elapsed = time.time() - START_TIME
    if elapsed > MAX_RUNTIME_SEC:
        print(f"\n[TIME] Reached max runtime ({elapsed:.0f}s). Saving progress.")
        break

    batch = words[i:i+BATCH_SIZE]
    response = classify_batch(batch)
    removed = parse_response(response)
    to_remove.update([w.lower() for w in removed])
    i += len(batch)
    print(f"[{i}/{total}] removed so far: {len(to_remove)} | elapsed: {elapsed:.0f}s")

    if (i // BATCH_SIZE) % 10 == 0:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "to_remove": sorted(to_remove),
                "processed": i,
            }, f)
        print(f"  [progress saved: {i}/{total}]")

# Save final progress
with open(PROGRESS_FILE, "w") as f:
    json.dump({
        "to_remove": sorted(to_remove),
        "processed": i,
    }, f)

print(f"\nProcessed: {i}/{total}")
print(f"To remove: {len(to_remove)}")

if i >= total:
    print("All words processed. Cleaning core.json...")
    cleaned = {}
    removed_count = 0
    for word, code in core.items():
        if word.startswith("__"):
            cleaned[word] = code
            continue
        if word.lower() in to_remove:
            removed_count += 1
            continue
        cleaned[word] = code

    print(f"Removed {removed_count} words")
    print(f"Core: {len(core)} -> {len(cleaned)}")

    with open(CORE_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print("Done.")
else:
    print(f"Partial run. {total - i} words remaining.")
