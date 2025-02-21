import cv2
import numpy as np
import hashlib
from scipy.fftpack import dct, idct
from PIL import Image
import imagehash
import os

### ✅ 1. SHA-256 해시 생성 ###
def generate_image_hash(file_path, hash_algorithm='sha256'):
    """이미지 파일의 해시를 생성"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 파일이 존재하지 않습니다: {file_path}")

    hash_func = getattr(hashlib, hash_algorithm, None)
    if hash_func is None:
        raise ValueError(f"지원되지 않는 해시 알고리즘: {hash_algorithm}")

    hasher = hash_func()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


### ✅ 2. DCT 변환 및 역변환 ###
def apply_dct(img):
    """DCT 변환"""
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def apply_idct(img):
    """IDCT 변환"""
    return idct(idct(img.T, norm='ortho').T, norm='ortho')


### ✅ 3. DCT 기반 워터마킹 삽입 ###
def embed_watermark(image_path, output_path, hash_algorithm='sha256', watermark_strength=150, offset=50):
    """DCT 변환 후 해시값을 워터마크로 삽입"""

    # 1️⃣ 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"❌ 이미지 로드 실패: {image_path}")

    h, w, c = img.shape

    # 2️⃣ 이미지 해시 생성
    image_hash = generate_image_hash(image_path, hash_algorithm)
    print(f"✅ 생성된 해시 값: {image_hash}")

    # 3️⃣ DCT 변환 수행
    dct_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        dct_img[:, :, i] = apply_dct(img[:, :, i].astype(np.float32))

    # 4️⃣ 해시 값을 텍스트로 변환하여 워터마크 삽입
    watermark = np.zeros((h, w), dtype=np.float32)
    text_bits = [int(b) for b in ''.join(f'{ord(c):08b}' for c in image_hash[:16])]  # 해시 앞 16자리만 삽입
    
    # 중앙에 워터마크 삽입 (변수화)
    for i, bit in enumerate(text_bits):
        x, y = divmod(i, w)
        if x + offset < h and y + offset < w:
            watermark[x + offset, y + offset] = bit * watermark_strength

    # 5️⃣ DCT 변환 이미지에 워터마크 추가
    watermarked_dct = dct_img + np.repeat(watermark[:, :, np.newaxis], c, axis=2)

    # 6️⃣ IDCT 변환 (이미지 복원)
    watermarked_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        watermarked_img[:, :, i] = apply_idct(watermarked_dct[:, :, i])

    # 7️⃣ 최종 이미지 저장 (PNG 무손실 저장)
    watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    final_output_path = output_path.replace('.jpeg', '.png')
    cv2.imwrite(final_output_path, watermarked_img)
    
    print(f"✅ 워터마크 삽입 완료! 저장 파일: {final_output_path}")


### ✅ 4. DCT 워터마크 복원 및 해시 비교 ###
def extract_watermark(image_path, offset=50, threshold=75):
    """DCT 변환을 통해 삽입된 해시 기반 워터마크 추출"""
    
    # 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"❌ 이미지 로드 실패: {image_path}")

    h, w, c = img.shape

    # DCT 변환 수행
    dct_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        dct_img[:, :, i] = apply_dct(img[:, :, i].astype(np.float32))

    # 워터마크 추출 (변수화)
    extracted_bits = []
    for i in range(128):  # 16문자 * 8비트 = 128비트 추출
        x, y = divmod(i, w)
        if x + offset < h and y + offset < w:
            bit = 1 if dct_img[x + offset, y + offset, 0] > threshold else 0
            extracted_bits.append(bit)

    # 바이너리 문자열을 ASCII 문자로 변환
    extracted_text = ''.join(chr(int(''.join(map(str, extracted_bits[i:i+8])), 2)) for i in range(0, len(extracted_bits), 8))

    print(f"🔹 추출된 해시 기반 워터마크: {extracted_text}")
    return extracted_text


### ✅ 5. pHash 비교 (Perceptual Hash) ###
def perceptual_hash(image_path):
    """Perceptual Hash (pHash) 생성"""
    image = Image.open(image_path).convert("RGB")
    return imagehash.phash(image)

def compare_perceptual_hashes(original_path, watermarked_path, threshold=5):
    """Perceptual Hash (pHash) 비교"""
    original_phash = perceptual_hash(original_path)
    watermarked_phash = perceptual_hash(watermarked_path)

    print(f"🔹 원본 pHash: {original_phash}")
    print(f"🔹 워터마킹된 pHash: {watermarked_phash}")

    difference = original_phash - watermarked_phash
    print(f"🔍 해시 차이(유사도): {difference}")

    if difference == 0:
        print("✅ 두 이미지는 동일합니다! (Perceptual Hash 일치)")
    elif difference <= threshold:
        print("⚠️ 약간의 차이가 있지만, 같은 이미지로 볼 수 있습니다.")
    else:
        print("❌ 이미지가 변경되었습니다!")


### ✅ 6. 실행 ###
image_path = "C:/SecurityMark/hash+watermark/skuman.jpg"
output_path = "watermarked_image.png"

# 워터마킹 삽입
embed_watermark(image_path, output_path)

# pHash 비교
compare_perceptual_hashes(image_path, output_path)

# 워터마크 복원
extracted_hash = extract_watermark(output_path)

# 원본 해시 비교
original_hash = generate_image_hash(image_path)[:16]
if extracted_hash == original_hash:
    print("✅ 워터마크가 정상적으로 복원되었습니다! (해시 일치)")
else:
    print("⚠️ 워터마크가 변조되었거나 손실되었습니다! (해시 불일치)")
