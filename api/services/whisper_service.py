import openai
from api.config import settings

# OpenAI API 설정
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

def transcribe_audio(file_path: str) -> dict:
    """
    OpenAI Whisper를 사용하여 음성 파일을 텍스트로 변환하는 함수.
    :param file_path: 변환할 음성 파일의 경로
    :return: 변환된 텍스트를 포함한 JSON 형식의 응답
    """
    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",  # OpenAI Whisper 모델
                file=audio_file,
                response_format="json"  # JSON 형식으로 응답
            )
            
        if not hasattr(response, "text"):  # 🚨 "text" 속성이 없는 경우
            return {"status": "error", "message": "Missing 'text' attribute in API response"}
        
        result = response.text  # 변환된 텍스트 추출
        return {"status": "success", "text": result}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

# # 예제 사용법
# if __name__ == "__main__":
#     file_path = "path/to/your/audio/file.mp3"  # 실제 파일 경로로 변경
#     transcription_result = transcribe_audio(file_path)
#     print(json.dumps(transcription_result, ensure_ascii=False, indent=4))
