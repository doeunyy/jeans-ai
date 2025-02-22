import json
import re
import pytz
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from openai import AsyncOpenAI
from api.config import settings


client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def process_photo_search(text: str, functions: list) -> dict:
    """
    사진 검색 기능을 처리하는 함수.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "사용자의 요청을 이해하고 적절한 사진 검색 기능을 호출하세요."},
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
        extracted_text = request_text

        date = arguments.get("date", "")
        keyword = arguments.get("keyword", "")
        target_name = arguments.get("target_name", None)

        kst = pytz.timezone("Asia/Seoul")
        today = datetime.now(kst).date()

        if re.match(r"\d{4}-\d{2}-\d{2}", date):
            date_parsed = date
            date_start = date
            date_end = None
        else:
            date_parsed, date_start, date_end = parse_relative_date(date) if date else ("", "", "")

        if date_end and datetime.strptime(date_end, "%Y-%m-%d").date() > today:
            date_end = today.strftime("%Y-%m-%d")

        return {
            # "text": extracted_text,
            # "function_call": function_call_result,
            # "date": date_parsed,
            "date_start": date_start,
            "date_end": date_end,
            "keyword": keyword,
            "target_name": target_name
        }

    except Exception as e:
        return {"error": str(e)}


# 상대적인 날짜 표현을 변환하는 함수 (date_start, date_end 포함)
def parse_relative_date(date_str: str) -> tuple:
    """
    사용자가 입력한 상대적인 날짜를 실제 날짜(YYYY-MM-DD)로 변환.
    - 기준 날짜(date): 사용자가 입력한 "일년 전", "한달 전" 등의 값을 변환한 날짜
    - date_start: 기준 날짜에서 2주 전
    - date_end: 기준 날짜에서 2주 후
    """
    kst = pytz.timezone("Asia/Seoul")  # 한국 시간대 설정
    today = datetime.now(kst).date()  # KST 기준 오늘 날짜

    # 정규식을 사용하여 상대적 날짜 패턴 추출
    match = re.search(r'(\d+)?\s*(일|주|개월|년) 전', date_str)
    if match:
        num = int(match.group(1)) if match.group(1) else 1  # 숫자가 없으면 기본값 1
        unit = match.group(2)

        # 날짜 계산
        if unit == "일":
            base_date = today - timedelta(days=num)
        elif unit == "주":
            base_date = today - timedelta(weeks=num)
        elif unit == "개월":
            base_date = today - relativedelta(months=num)
        elif unit == "년":
            base_date = today - relativedelta(years=num)
        else:
            return date_str, "", ""  # 변환 불가하면 원본 반환

        # 기준 날짜로부터 2주 전, 2주 후 계산
        date_start = base_date - timedelta(weeks=2)
        date_end = base_date + timedelta(weeks=2)

        return base_date.strftime("%Y-%m-%d"), date_start.strftime("%Y-%m-%d"), date_end.strftime("%Y-%m-%d")

    return date_str, "", ""  # 변환이 불가능한 경우 원본 문자열 그대로 반환
