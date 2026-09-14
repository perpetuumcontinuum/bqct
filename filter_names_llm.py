import json
import os
import re
from llama_cpp import Llama

MODEL_PATH = "models/Qwen3-14B-Q5_K_M.gguf"
CORE_FILE = "core.json"
NAMES_FILE = "names_to_filter.txt"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))

print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=4,
    verbose=False,
)

# Load words to filter
with open(NAMES_FILE, "r", encoding="utf-8") as f:
    words = [line.strip().lower() for line in f if line.strip()]

print(f"Words to classify: {len(words)}")

# Load core.json
with open(CORE_FILE, "r", encoding="utf-8") as f:
    core = json.load(f)

print(f"Core: {len(core)} entries")

SYSTEM = """You are a linguistic classifier for a minimal constructed language.

For each English word, decide:
- KEEP if it is a common noun, verb, adjective, or adverb (ordinary word).
- REMOVE if it is a proper name, family name, brand, or place name.

Examples:
- KEEP: love, rose, king, sun, may, will, peace, faith, grace, joy, price, power
- REMOVE: john, mary, smith, kardashian, obama, whatsapp, instagram, minecraft

If uncertain, KEEP.

Respond ONLY with valid JSON, no explanation:
{"keep": ["word1", "word2"], "remove": ["word3", "word4"]}"""

def classify_batch(batch):
    prompt = "Words: " + ", ".join(batch)
    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
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

to_remove = set()
total = len(words)

for i in range(0, total, BATCH_SIZE):
    batch = words[i:i+BATCH_SIZE]
    keep, remove = classify_batch(batch)
    to_remove.update([w.lower() for w in remove])
    print(f"[{min(i+BATCH_SIZE, total)}/{total}] kept={len(keep)} removed={len(remove)}")

print(f"\nTotal to remove: {len(to_remove)}")

# Clean core.json
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

print(f"Saved to {CORE_FILE}")
