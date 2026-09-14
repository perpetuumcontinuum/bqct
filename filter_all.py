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

SYSTEM = """You are a linguistic classifier for a minimal constructed language.

For each English word, decide:
- KEEP if it is a common noun, verb, adjective, or adverb (ordinary word).
- REMOVE if it is a proper name, family name, brand, place name, or an obvious typo.

Examples:
- KEEP: love, rose, king, sun, may, will, peace, faith, grace, joy, price, power
- REMOVE: john, mary, smith, kardashian, obama, whatsapp, instagram, minecraft, consious

If uncertain, KEEP.

Respond ONLY with valid JSON, no explanation:
{"keep": ["word1", "word2"], "remove": ["word3", "word4"]}"""

print("Warming up (caching system prompt)...")
llm.create_chat_completion(
    messages=[{"role": "system", "content": SYSTEM}],
    max_tokens=1,
    temperature=0.0,
)

with open(CORE_FILE, "r", encoding="utf-8") as f:
    core = json.load(f)

words = [w for w in core.keys() if not w.startswith("__")]
total = len(words)
print(f"Words to classify: {total}")

def classify_batch(batch):
    prompt = "Words: " + ", ".join(batch)
    try:
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.0,
        )
        text = result["choices"][0]["message"]["content"]
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return [], []
        data = json.loads(m.group(0))
        return data.get("keep", []), data.get("remove", [])
    except Exception as e:
        print(f"  [ERROR] {e}")
        return [], []

i = RESUME_FROM
while i < total:
    # Check runtime
    elapsed = time.time() - START_TIME
    if elapsed > MAX_RUNTIME_SEC:
        print(f"\n[TIME] Reached max runtime ({elapsed:.0f}s). Saving progress and exiting.")
        break

    batch = words[i:i+BATCH_SIZE]
    keep, remove = classify_batch(batch)
    to_remove.update([w.lower() for w in remove])
    i += len(batch)
    print(f"[{i}/{total}] removed so far: {len(to_remove)} | elapsed: {elapsed:.0f}s")

    # Save progress every 10 batches
    if (i // BATCH_SIZE) % 10 == 0:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "to_remove": sorted(to_remove),
                "processed": i,
            }, f)
        print(f"  [progress saved: {i}/{total}]")

# Always save final progress
with open(PROGRESS_FILE, "w") as f:
    json.dump({
        "to_remove": sorted(to_remove),
        "processed": i,
    }, f)

print(f"\nProcessed: {i}/{total}")
print(f"To remove: {len(to_remove)}")

# Only clean core.json if we processed ALL words
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
    print("Progress saved. Next run will resume automatically.")
