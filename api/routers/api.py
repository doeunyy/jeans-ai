import json
import openai
import tempfile
from fastapi import APIRouter, WebSocket
from pydantic import BaseModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from api.services import gpt_service, whisper_service
from api.functions.function_registry import get_function_list
from api.config import settings

router = APIRouter()
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

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
    
### 사진 댓글용 
@router.post("/text")
async def process_audio_to_text(request: AudioRequest) -> dict:
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

#### WebSocket - STT
@router.websocket("/ws-text")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket 클라이언트 연결됨!")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                print("🔍 [DEBUG] 받은 데이터 없음. 연결 종료")
                break

            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
                temp_audio.write(data)
                temp_audio_path = temp_audio.name
            print(f"✅ 음성 파일 저장 완료: {temp_audio_path}")

            # Whisper 변환
            whisper_response = whisper_service.transcribe_audio(temp_audio_path)
            
            # 클라이언트에게 텍스트 전송
            await websocket.send_text(whisper_response)
            print("✅ 변환된 텍스트 전송 완료!")

    except Exception as e:
        print(f"❌ WebSocket 오류 발생: {e}")
    finally:
        await websocket.close()
        
### WebSocket - GPT Function call
@router.websocket("/ws-process")
async def websocket_audio_to_function(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket 클라이언트 연결됨!")

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                print("🔍 [DEBUG] 받은 데이터 없음. 연결 종료")
                break

            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
                temp_audio.write(data)
                temp_audio_path = temp_audio.name
            print(f"✅ 음성 파일 저장 완료: {temp_audio_path}")

            # Whisper 변환
            whisper_response = whisper_service.transcribe_audio(temp_audio_path)
            
            # GPT를 사용한 추가 처리 수행
            functions = get_function_list()
            gpt_response = await gpt_service.process_with_functions(whisper_response, functions)
            
            # WebSocket 전송 전에 JSON 문자열로 변환
            if isinstance(gpt_response, dict):  
                result = json.dumps(gpt_response)  # JSON 문자열 변환
            
            # 클라이언트에게 텍스트 전송
            await websocket.send_text(result)
            print("✅ 변환된 텍스트 전송 완료!")

    except Exception as e:
        print(f"❌ WebSocket 오류 발생: {e}")
    finally:
        await websocket.close()


### 파인튜닝된 모델 로드
# MODEL_PATH = "./whisper/model_finetuned/whisper-small-2025-02-23_1123"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, processor, _ = whisper_service.load_finetuned_model(MODEL_PATH, device)

# @router.post("/process")
# async def process_audio_to_function(request: AudioRequest) -> dict:
#     """
#     파인튜닝된 Whisper 모델을 사용하여 음성 파일을 텍스트로 변환한 후 GPT 처리로 전달하는 함수.
#     :param request: 변환할 음성 파일 경로가 포함된 요청
#     :return: 변환된 텍스트를 포함한 JSON 형식의 응답
#     """
#     try:
#         # Whisper를 사용한 변환 수행 (파인튜닝된 모델 사용)
#         whisper_response = whisper_service.transcribe_with_finetuned_model(model, processor, request.file_path, device)

#         if whisper_response["status"] != "success":
#             return whisper_response  # 오류 발생 시 바로 반환

#         result = whisper_response["text"]
#         print(f"🔍 [DEBUG] 변환된 텍스트: {result}")

#         # GPT를 사용한 추가 처리 수행
#         functions = get_function_list()
#         gpt_response = await gpt_service.process_with_functions(result, functions)

#         return gpt_response

#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# @router.post("/text")
# async def process_audio_to_text(request: AudioRequest) -> dict:
#     """
#     파인튜닝된 Whisper 모델을 사용하여 음성 파일을 텍스트로 변환한 후 GPT 처리로 전달하는 함수.
#     :param request: 변환할 음성 파일 경로가 포함된 요청
#     :return: 변환된 텍스트를 포함한 JSON 형식의 응답
#     """
#     try:
#         # Whisper를 사용한 변환 수행 (파인튜닝된 모델 사용)
#         whisper_response = whisper_service.transcribe_with_finetuned_model(model, processor, request.file_path, device)

#         if whisper_response["status"] != "success":
#             return whisper_response  # 오류 발생 시 바로 반환

#         result = whisper_response["text"]
#         return result

#     except Exception as e:
#         return {"status": "error", "message": str(e)}