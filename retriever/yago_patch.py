import os

HEADER_FILE = "data/yago/beyond_aa"
TARGET_DIR = "data/yago"

# 读取标准前缀头部
with open(HEADER_FILE, "r") as f:
    header_lines = []
    for line in f:
        if not line.strip().startswith("@prefix"):
            break
        header_lines.append(line)
HEADER = "".join(header_lines)

print(f"[INFO] Extracted {len(header_lines)} prefix lines from {HEADER_FILE}")

patched = 0
for name in sorted(os.listdir(TARGET_DIR)):
    path = os.path.join(TARGET_DIR, name)
    if not os.path.isfile(path) or name.endswith("_aa"):
        continue

    with open(path, "r") as f:
        content = f.read()

    # Skip if already has "yago:" prefix
    if "@prefix yago:" in content:
        continue

    with open(path, "w") as f:
        f.write(HEADER + "\n" + content)
    patched += 1
    print(f"[PATCHED] {name}")

print(f"\n✅ Patched {patched} files.")