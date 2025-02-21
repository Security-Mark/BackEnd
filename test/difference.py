import cv2
import numpy as np

def show_image_difference(original_path, processed_path):
    """ 원본 이미지와 패턴 삽입된 이미지 간 차이 시각화 """
    
    # 이미지 로드 (그레이스케일 변환)
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

    if original is None or processed is None:
        raise ValueError("❌ 이미지 로드 실패! 경로를 확인하세요.")

    # 차이 이미지 계산
    diff = cv2.absdiff(original, processed)

    # 차이를 강조 (255 범위로 정규화)
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 결과 출력
    cv2.imshow("Difference Image", diff_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 차이 이미지 저장
    cv2.imwrite("final_difference_image.png", diff_enhanced)
    print("✅ 차이 이미지가 'final_difference_image.png'로 저장되었습니다.")

# 실행
show_image_difference("watermarked_image.png", "final_anti_deepfake_image.png")
