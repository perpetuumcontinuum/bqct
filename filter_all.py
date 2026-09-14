import json
import os
import re
import time
from llama_cpp import Llama

MODEL_PATH = "models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
CORE_FILE = "core.json"
PROGRESS_FILE = "progress.json"

BATCH_SIZE = 20
CONFIDENCE_THRESHOLD = 70
N_THREADS = os.cpu_count() or 4
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "19000"))
START_TIME = time.time()

print(f"Threads: {N_THREADS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}%")
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
        "For each word, answer: word KEEP/REMOVE certainty\n"
        "Certainty is 0-100.\n"
        "If unsure, still give your best guess.\n\n"
        "Examples:\n"
        "love KEEP 95\n"
        "john REMOVE 90\n"
        "rose KEEP 95\n"
        "kardashian REMOVE 95\n"
        "googoo REMOVE 80\n"
        "smith KEEP 85\n\n"
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

def parse_response(text, batch, threshold=70):
    remove = []
    batch_set = set(w.lower() for w in batch)
    answered = set()
    for line in text.splitlines():
        line = line.strip().lower()
        if not line:
            continue
        m = re.match(r'^([a-z]+)\s+(keep|remove)\s+(\d+)', line)
        if m:
            word, verdict, conf = m.group(1), m.group(2), int(m.group(3))
            if word in batch_set:
                answered.add(word)
                if verdict == "remove" and conf >= threshold:
                    remove.append(word)
    missing = batch_set - answered
    if missing:
        print(f"  [WARN] Qwen missed {len(missing)} words: {sorted(missing)[:5]}")
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
    removed = parse_response(response, batch, CONFIDENCE_THRESHOLD)
    if removed:
        print(f"  Removed (>= {CONFIDENCE_THRESHOLD}%): {removed}")
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
    removed_dict = {}
    removed_count = 0
    for word, code in core.items():
        if word.startswith("__"):
            cleaned[word] = code
            continue
        if word.lower() in to_remove:
            removed_dict[word] = code
            removed_count += 1
            continue
        cleaned[word] = code

    print(f"Removed {removed_count} words")
    print(f"Core: {len(core)} -> {len(cleaned)}")

    with open(CORE_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Saved cleaned core to {CORE_FILE}")

    REMOVED_FILE = "removed.json"
    with open(REMOVED_FILE, "w", encoding="utf-8") as f:
        json.dump(removed_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved removed words to {REMOVED_FILE}")

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print("Done.")
else:
    print(f"Partial run. {total - i} words remaining.")
