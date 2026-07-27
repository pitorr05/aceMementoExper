import json
import os
import math
import argparse

def update_qwen_config(model_path, target_context=163840):
    config_file = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_file):
        print(f"❌ Không tìm thấy file tại: {config_file}")
        return

    # 1. Đọc dữ liệu hiện tại
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. Lấy thông số gốc
    # Ưu tiên đọc từ bên trong rope_scaling nếu đã có, rồi mới đọc top-level
    existing_rope = config.get("rope_scaling", {})
    orig_max = (
        existing_rope.get("original_max_position_embeddings")
        or config.get("original_max_position_embeddings")
        or config.get("max_position_embeddings", 32768)
    )
    
    # 3. Tính toán factor
    # Công thức: factor = target / original
    raw_factor = target_context / orig_max
    # Làm tròn lên 1 chữ số thập phân để an toàn (ví dụ 4.58 -> 4.6 hoặc 5.0)
    factor = math.ceil(raw_factor * 10) / 10 
    
    print(f"🔄 Đang cấu hình: {orig_max} -> {target_context} (Factor: {factor})")

    # 4. Cập nhật hoặc thêm mới rope_scaling
    # Dùng "rope_type" (transformers >= 4.40) VÀ "type" để tương thích ngược
    config["rope_scaling"] = {
        "rope_type": "yarn",
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": orig_max,
    }
    
    # Đảm bảo max_position_embeddings cũng được cập nhật để vLLM nhận diện đúng
    config["max_position_embeddings"] = target_context

    # 5. Lưu lại file
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã cập nhật xong file: {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./model")
    parser.add_argument("--target-context", type=int, default=163840)
    args = parser.parse_args()

    update_qwen_config(model_path=args.model_path, target_context=args.target_context)