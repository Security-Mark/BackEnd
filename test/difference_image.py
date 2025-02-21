import cv2
import numpy as np

def show_image_difference(original_path, watermarked_path):
    """ 원본 이미지와 워터마킹된 이미지 간 차이 시각화 """
    
    # 이미지 로드 (그레이스케일 변환)
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)

    if original is None or watermarked is None:
        raise ValueError("❌ 이미지 로드 실패! 경로를 확인하세요.")

    # 차이 이미지 계산
    diff = cv2.absdiff(original, watermarked)

    # 차이를 강조 (255 범위로 정규화)
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 결과 출력
    cv2.imshow("Difference Image", diff_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 차이 이미지 저장
    cv2.imwrite("difference_image.png", diff_enhanced)
    print("✅ 차이 이미지가 'difference_image.png'로 저장되었습니다.")

# 실행
show_image_difference("C:/SecurityMark/hash+watermark/skuman.jpg", "watermarked_image.png")
