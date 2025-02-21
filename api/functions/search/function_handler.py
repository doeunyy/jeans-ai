async def handle_search_photo(params: dict) -> dict:
    date = params.get("date")
    keyword = params.get("keyword")
    user = params.get("user", "없음")  # 사용자 이름은 선택적

    # 실제 사진 검색 로직
    # 예시: 외부 API 호출 또는 데이터베이스 쿼리로 검색 결과를 얻어냄
    result = [
        f"{date}에 촬영된 {keyword} 사진1",
        f"{date}에 촬영된 {keyword} 사진2",
        f"{date}에 촬영된 {keyword} 사진3"
    ]
    
    return {
        "status": "success",
        "result": result
    }