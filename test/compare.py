import hashlib

def generate_image_hash(file_path, hash_algorithm='sha256'):
    """이미지 파일의 해시를 생성"""
    hash_func = getattr(hashlib, hash_algorithm, None)
    if hash_func is None:
        raise ValueError(f"지원되지 않는 해시 알고리즘: {hash_algorithm}")

    hasher = hash_func()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

def compare_image_hashes(original_path, processed_path):
    """ 원본 이미지와 변형된 이미지의 해시 비교 """
    original_hash = generate_image_hash(original_path)
    processed_hash = generate_image_hash(processed_path)

    print(f"🔹 원본 이미지 해시: {original_hash}")
    print(f"🔹 변형된 이미지 해시: {processed_hash}")

    if original_hash == processed_hash:
        print("✅ 해시가 동일합니다. 패턴이 비가시적으로 삽입되었습니다.")
    else:
        print("⚠️ 해시가 다릅니다! 패턴이 삽입되어 변경되었습니다.")

# 실행
compare_image_hashes("watermarked_image.png", "final_anti_deepfake_image.png")
