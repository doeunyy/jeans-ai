import re
import json
from openai import AsyncOpenAI
from api.config import settings


client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def process_delete_friend(text: str, functions: list) -> dict:
    """
    친구 삭제 기능을 처리하는 함수.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "사용자의 요청을 이해하고 친구 삭제 기능을 실행하세요."},
                {"role": "user", "content": text}
            ],
            functions=functions,
            function_call="auto"
        )

        function_call_result = response.choices[0].message.function_call

        if hasattr(function_call_result, 'arguments') and function_call_result.arguments:
            arguments = json.loads(function_call_result.arguments)
        else:
            arguments = {}

        request_text = arguments.get("request", text)
        
        # 친구 이름 뒤의 "을" 제거
        match = re.search(r"([\w가-힣]+)을", request_text)
        if match:
            target_name = match.group(1)
        else:
            target_name = request_text.split(" ")[0] if request_text else None

        return {
            # "text": text,
            # "function_call": function_call_result,
            "target_name": target_name,
        }

    except Exception as e:
        return {"error": str(e)}