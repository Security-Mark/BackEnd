import imagehash
from PIL import Image
import hashlib

def perceptual_hash(image_path):
    """Perceptual Hash (pHash) 생성"""
    image = Image.open(image_path).convert("RGB")
    return imagehash.phash(image)

def compare_perceptual_hashes(original_path, processed_path):
    """Perceptual Hash (pHash) 비교"""
    original_phash = perceptual_hash(original_path)
    processed_phash = perceptual_hash(processed_path)

    print(f"🔹 원본 pHash: {original_phash}")
    print(f"🔹 처리된 이미지 pHash: {processed_phash}")

    difference = original_phash - processed_phash
    print(f"🔍 pHash 차이(유사도): {difference}")

    if difference == 0:
        print("✅ 두 이미지는 육안으로 동일합니다! (Perceptual Hash 일치)")
    elif difference <= 5:
        print("⚠️ 약간의 차이가 있지만, 시각적으로는 거의 동일합니다.")
    else:
        print("❌ 원본과 차이가 발생하였습니다. 패턴 강도를 줄여야 합니다.")

# 실행 (원본 이미지 vs 패턴 적용 이미지)
compare_perceptual_hashes("watermarked_image.png", "final_anti_deepfake_image.png")
