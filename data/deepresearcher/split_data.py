# split_data.py - Tạo ở thư mục gốc, chạy 1 lần
import json
import random
from pathlib import Path

# Đọc toàn bộ dữ liệu
with open(r"C:\Users\Admin\ACEmento\data\deepresearcher\deepresearcher.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"Total samples: {len(samples)}")

# Shuffle để trộn dữ liệu
random.seed(42)
random.shuffle(samples)

# Tỷ lệ: 70% train, 15% val, 15% test
train_split = int(0.7 * len(samples))
val_split = int(0.85 * len(samples))

train_data = samples[:train_split]
val_data = samples[train_split:val_split]
test_data = samples[val_split:]

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Tạo thư mục
Path("data/deepresearcher").mkdir(parents=True, exist_ok=True)

# Ghi ra file
for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
    with open(f"data/deepresearcher/{name}.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f" Created data/deepresearcher/{name}.jsonl")