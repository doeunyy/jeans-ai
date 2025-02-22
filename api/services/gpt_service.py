from openai import AsyncOpenAI
from api.config import settings
from api.services.function_registry import get_function_handler, get_front_path


# OpenAI API 설정
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def process_with_functions(text: str, functions: list) -> dict:
    try:
        function_type = await classify_function(text)   # 기능 판별
        handler = get_function_handler(function_type)   # 기능 핸들러 가져오기
        action, front_path = get_front_path(function_type)      # 기능 path 가져오기
        
        print(f"🔍 [DEBUG] 기능 판별 결과: {function_type}, Front Path: {front_path}")  # 디버깅 로그
        
        if handler:
            response = await handler(text, functions)   # 해당 기능 실행
            response["action"] = action                 # action 추가                
            response["path"] = front_path               # path 추가
            return response
        elif front_path:  # 핸들러가 없어도 front_path가 있으면 반환
            return {"action": action, "path": front_path}
        else:
            return {"error": f"'{function_type}'에 대한 처리 핸들러 또는 경로가 없습니다."}

    except Exception as e:
        return {"error": str(e)}



async def classify_function(text: str) -> str:
    """
    GPT를 이용해 입력된 문장이 어떤 기능을 요청하는지 판단.
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "사용자의 요청이 어떤 기능인지 분석하세요."
                            "아래의 기능 중 하나의 단어만 반환하세요.\n"
                            "- 친구 목록 조회: view_friends_list\n"
                            "- 친구 삭제: delete_friend\n"
                            "- 친구 요청 수락: accept_friend_request\n"
                            "- 사진 검색: search_photo\n" 
                            #! 여기에 계속 추가해야함 
                            "반환할 때 단어 하나만 반환하세요. "
            },
            {"role": "user", "content": text}
        ]
    )

    result = response.choices[0].message.content.strip()

    # 따옴표 제거
    result = result.replace("'", "").replace('"', "").strip()

    print(f"🔍 [DEBUG] GPT API 응답: {result}")  # ✅ 디버깅 로그 추가

    return result