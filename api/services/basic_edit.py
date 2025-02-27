import io
import os
import requests
from PIL import Image, ImageEnhance


# 보정 값 정의
exposure_high = 1.2
exposure_low = 0.8
saturation_high = 1.3
saturation_low = 0.7
brightness_high = 1.2
brightness_low = 0.8

# 저장될 이미지 경로 설정
STATIC_DIR = "/home/ubuntu/fastapi/static/images"
# BASE_URL = "http://13.124.56.47:8001/static/images" # 테스트용
BASE_URL = "https://api.passion4-jeans-ai.store/static/images"

# 저장 폴더가 없으면 생성
os.makedirs(STATIC_DIR, exist_ok=True)

# 보정 케이스별 함수
def apply_adjustments(image, exposure_factor, saturation_factor, brightness_factor):
    """ 노출, 채도, 밝기 조정을 적용하는 함수 """
    enhancer_exposure = ImageEnhance.Brightness(image)
    image = enhancer_exposure.enhance(exposure_factor)

    enhancer_saturation = ImageEnhance.Color(image)
    image = enhancer_saturation.enhance(saturation_factor)

    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness_factor)
    
    return image

def select_case(image, apply_exposure, apply_saturation, apply_brightness):
    case_index = (apply_exposure << 2) | (apply_saturation << 1) | apply_brightness
    
    cases = {
        0b000: (exposure_low, saturation_low, brightness_low),
        0b001: (exposure_low, saturation_low, brightness_high),
        0b010: (exposure_low, saturation_high, brightness_low),
        0b011: (exposure_low, saturation_high, brightness_high),
        0b100: (exposure_high, saturation_low, brightness_low),
        0b101: (exposure_high, saturation_low, brightness_high),
        0b110: (exposure_high, saturation_high, brightness_low),
        0b111: (exposure_high, saturation_high, brightness_high),
    }

    exposure_factor, saturation_factor, brightness_factor = cases[case_index]
    return apply_adjustments(image, exposure_factor, saturation_factor, brightness_factor)

def apply_basic_edit(image_url: str, apply_grayscale: bool, apply_contrast: bool, apply_sharpen: bool) -> str:
    ''' 
    이미지 URL을 받아 보정을 적용하고 파일로 저장 
    '''
    
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("이미지를 다운로드할 수 없습니다.")

    image = Image.open(io.BytesIO(response.content))

    # 추가적인 노출, 채도, 밝기 보정 적용
    image = select_case(image, apply_grayscale, apply_contrast, apply_sharpen)

    # # 결과 저장 경로
    # output_path = "edited_image.jpg"
    # image.save(output_path)
    # return output_path
    
    # 저장할 파일명 생성 (유니크한 이름 생성)
    file_name = "edited_image.jpg"
    file_path = os.path.join(STATIC_DIR, file_name)

    # 이미지 저장
    image.save(file_path)

    # 저장된 이미지의 URL 반환
    return f"{BASE_URL}/{file_name}"
