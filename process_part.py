import json
import os
import re
import sys
from llama_cpp import Llama

MODEL_PATH = "models/Qwen3-14B-Q5_K_M.gguf"
CORE_FILE = "core.json"
PART = int(os.environ.get("PART", "0"))
TOTAL_PARTS = int(os.environ.get("TOTAL_PARTS", "20"))
BATCH_SIZE = 50

print(f"Part {PART}/{TOTAL_PARTS}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=4,
    verbose=False,
)

with open(CORE_FILE, "r", encoding="utf-8") as f:
    core = json.load(f)

# Filter reserved
words = [w for w in core.keys() if not w.startswith("__")]

# Take my slice
chunk_size = len(words) // TOTAL_PARTS + 1
start = PART * chunk_size
end = min(start + chunk_size, len(words))
my_words = words[start:end]

print(f"My words: {len(my_words)} ({start}..{end})")

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

for i in range(0, len(my_words), BATCH_SIZE):
    batch = my_words[i:i+BATCH_SIZE]
    keep, remove = classify_batch(batch)
    to_remove.update([w.lower() for w in remove])
    print(f"[{min(i+BATCH_SIZE, len(my_words))}/{len(my_words)}] removed={len(remove)}")

print(f"\nTotal to remove: {len(to_remove)}")

# Save result as artifact
result = {
    "part": PART,
    "to_remove": sorted(to_remove),
}
out = f"part_{PART:02d}_result.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved to {out}")
