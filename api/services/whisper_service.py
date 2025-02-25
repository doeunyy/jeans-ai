import os
import openai
import torch
import torchaudio
from api.config import settings
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# OpenAI API 설정
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

# m4a -> wav 변환
def convert_m4a_to_wav(m4a_path):
    wav_path = m4a_path.replace(".m4a", ".wav")
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")
    return wav_path

# OpenAI whisper 음성 -> 텍스트 변환
def transcribe_audio(audio_path: str) -> str:
    """
    음성 파일을 OpenAI Whisper API를 사용하여 텍스트로 변환.
    
    :param audio_path: 변환할 음성 파일의 경로
    :return: 변환된 텍스트
    """
    try:
        # m4a → wav 변환
        wav_path = convert_m4a_to_wav(audio_path)
        
        # Whisper API 호출
        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

        # 변환된 텍스트 반환
        whisper_response = transcription.text
        print(f"🔍 [DEBUG] 변환된 텍스트: {whisper_response}")

        return whisper_response

    except Exception as e:
        print(f"❌ Whisper 변환 실패: {e}")
        return "Error - Whisper 변환 실패"

    finally:
        # 변환된 wav 파일 삭제
        if os.path.exists(wav_path):
            os.remove(wav_path)


### 기존 버전
# def transcribe_audio(file_path: str) -> dict:
#     """
#     OpenAI Whisper를 사용하여 음성 파일을 텍스트로 변환하는 함수.
#     :param file_path: 변환할 음성 파일의 경로
#     :return: 변환된 텍스트를 포함한 JSON 형식의 응답
#     """
#     try:
#         with open(file_path, "rb") as audio_file:
#             response = client.audio.transcriptions.create(
#                 model="whisper-1",  # OpenAI Whisper 모델
#                 file=audio_file,
#                 response_format="json"  # JSON 형식으로 응답
#             )
            
#         if not hasattr(response, "text"):  # "text" 속성이 없는 경우
#             return {"status": "error", "message": "Missing 'text' attribute in API response"}
        
#         result = response.text  # 변환된 텍스트 추출
#         return {"status": "success", "text": result}
    
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


### 파인튜닝 모델 로드
def load_finetuned_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    저장된 파인튜닝 모델을 로드하는 함수.
    :param model_path: 저장된 모델 경로
    :param device: 사용할 디바이스 (기본값: CUDA 사용 가능하면 "cuda", 아니면 "cpu")
    :return: 로드된 모델과 프로세서
    """
    # 모델과 프로세서 로드
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    
    return model, processor, device

def transcribe_with_finetuned_model(model, processor, audio_path: str, device: str):
    """
    파인튜닝된 Whisper 모델을 사용하여 음성을 텍스트로 변환하는 함수.
    :param model: 로드된 Whisper 모델
    :param processor: Whisper 프로세서
    :param audio_path: 변환할 음성 파일 경로
    :param device: 실행할 디바이스 (CPU 또는 CUDA)
    :return: 변환된 텍스트
    """
    try:
        # 오디오 파일 로드 및 샘플링 레이트 조정
        waveform, sample_rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        # 오디오 데이터를 모델이 처리할 수 있는 입력으로 변환
        input_features = processor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # 텍스트 변환 수행
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return {"status": "success", "text": transcription}

    except Exception as e:
        return {"status": "error", "message": str(e)}