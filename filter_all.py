import json
import os
import re
import sys
from llama_cpp import Llama

MODEL_PATH = "models/qwen2.5-3b-instruct-q5_k_m.gguf"
CORE_FILE = "core.json"
PROGRESS_FILE = "progress.json"

BATCH_SIZE = 200
N_THREADS = os.cpu_count() or 4
RESUME_FROM = int(os.environ.get("RESUME_FROM", "0"))

print(f"Threads: {N_THREADS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Resume from: {RESUME_FROM}")

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

# Warmup: cache the system prompt
print("Warming up (caching system prompt)...")
llm.create_chat_completion(
    messages=[{"role": "system", "content": SYSTEM}],
    max_tokens=1,
    temperature=0.0,
)

# Load core.json
with open(CORE_FILE, "r", encoding="utf-8") as f:
    core = json.load(f)

words = [w for w in core.keys() if not w.startswith("__")]
total = len(words)
print(f"Words to classify: {total}")

# Load progress
to_remove = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        prog = json.load(f)
        to_remove = set(prog.get("to_remove", []))
        saved_resume = prog.get("processed", 0)
        if saved_resume > RESUME_FROM:
            RESUME_FROM = saved_resume
        print(f"Resumed: {len(to_remove)} to remove, processed={RESUME_FROM}")

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

# Process
i = RESUME_FROM
while i < total:
    batch = words[i:i+BATCH_SIZE]
    keep, remove = classify_batch(batch)
    to_remove.update([w.lower() for w in remove])
    i += len(batch)
    print(f"[{i}/{total}] removed so far: {len(to_remove)}")

    # Save progress every 10 batches
    if (i // BATCH_SIZE) % 10 == 0:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "to_remove": sorted(to_remove),
                "processed": i,
            }, f)
        print(f"  [progress saved: {i}/{total}]")

# Final clean
print(f"\nTotal to remove: {len(to_remove)}")
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
