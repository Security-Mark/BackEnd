import cv2
import numpy as np
from scipy.fftpack import dct, idct

def apply_dct(img):
    """DCT 변환"""
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def apply_idct(img):
    """IDCT 변환"""
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def embed_anti_deepfake_pattern(image_path, output_path, dct_strength=5, fourier_intensity=3, noise_intensity=2):
    """ 딥페이크 방어를 위한 DCT + Fourier + Adversarial Noise 패턴 삽입 """

    # 1️⃣ 이미지 로드 (컬러 이미지 유지)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"❌ 이미지 로드 실패: {image_path}")

    h, w, c = img.shape

    # 2️⃣ DCT 변형을 사용한 미세한 패턴 삽입
    dct_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        dct_img[:, :, i] = apply_dct(img[:, :, i].astype(np.float32))

    np.random.seed(42)
    dct_pattern = np.random.normal(0, dct_strength, (h, w))

    for i in range(c):
        dct_img[:, :, i] += dct_pattern  # RGB 채널 유지

    # 3️⃣ IDCT 변환 (이미지 복원)
    watermarked_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        watermarked_img[:, :, i] = apply_idct(dct_img[:, :, i])

    # 4️⃣ Fourier Transform 변형 적용
    transformed_channels = []
    for i in range(c):
        dft = np.fft.fft2(watermarked_img[:, :, i])
        dft_shift = np.fft.fftshift(dft)

        np.random.seed(42)
        fourier_pattern = np.random.normal(0, fourier_intensity, (h, w))
        dft_shift += fourier_pattern  # 주파수 변형 추가

        idft_shift = np.fft.ifftshift(dft_shift)
        transformed_channel = np.fft.ifft2(idft_shift)
        transformed_channels.append(np.abs(transformed_channel))

    watermarked_img = cv2.merge(transformed_channels)

    # 5️⃣ Adversarial Noise 삽입
    noise = np.random.normal(0, noise_intensity, img.shape).astype(np.float32)
    final_img = np.clip(watermarked_img + noise, 0, 255).astype(np.uint8)

    # 6️⃣ 이미지 저장
    cv2.imwrite(output_path, final_img)
    print(f"✅ 딥페이크 방어 패턴 삽입 완료! 저장 파일: {output_path}")

# 실행
embed_anti_deepfake_pattern("watermarked_image.png", "final_anti_deepfake_image.png")
