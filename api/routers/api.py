from api.config import settings
from fastapi import APIRouter
from pydantic import BaseModel
from api.services import gpt_service, whisper_service
from api.functions.function_registry import get_function_list

router = APIRouter()

class AudioRequest(BaseModel):
    file_path: str  # JSON 본문에서 파일 경로를 받기 위한 모델
    
@router.post("/process")
async def process_audio_to_function(request: AudioRequest) -> dict:
    """
    OpenAI Whisper를 사용하여 음성 파일을 텍스트로 변환한 후 GPT 처리로 전달하는 함수.
    :param file_path: 변환할 음성 파일의 경로
    :return: 변환된 텍스트를 포함한 JSON 형식의 응답
    """
    try:
        whisper_response = whisper_service.transcribe_audio(request.file_path)  # Whisper를 사용한 변환 수행
        if whisper_response["status"] != "success":
            return whisper_response  # 오류 발생 시 바로 반환
        
        # print(f"🔍 [DEBUG] whisper_response: {whisper_response}")  # 응답 값 확인
        # print(f"🔍 [DEBUG] whisper_response 타입: {type(whisper_response)}")  # 타입 확인
        
        result = whisper_response["text"]
        # print(f"🔍 [DEBUG] 변환된 텍스트: {result}")

        # GPT를 사용한 추가 처리 수행
        functions = get_function_list()
        gpt_response = await gpt_service.process_with_functions(result, functions)
        
        return gpt_response
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@router.post("/text")
async def process_audio_to_function(request: AudioRequest) -> dict:
    """
    OpenAI Whisper를 사용하여 음성 파일을 텍스트로 변환 후 전달
    """
    try:
        whisper_response = whisper_service.transcribe_audio(request.file_path)  # Whisper를 사용한 변환 수행
        if whisper_response["status"] != "success":
            return whisper_response  # 오류 발생 시 바로 반환
        
        # print(f"🔍 [DEBUG] whisper_response: {whisper_response}")  # 응답 값 확인
        # print(f"🔍 [DEBUG] whisper_response 타입: {type(whisper_response)}")  # 타입 확인
        
        result = whisper_response["text"]
        # print(f"🔍 [DEBUG] 변환된 텍스트: {result}")
        
        if isinstance(result, str):
            result = {"result": result}  # ✅ 문자열을 JSON으로 변환

        return result  # ✅ 이제 gpt_response는 딕셔너리이므로 오류 해결
    
    except Exception as e:
        return {"status": "error", "message": str(e)}