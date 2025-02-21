async def handle_delete_friend(params: dict) -> dict:
    target_name = params.get("target_name")
    # Spring Boot 백엔드로 요청을 전달하는 로직 (예시로 가정)
    result = {"status": "success", "message": f"{target_name} 친구 삭제 완료"}
    return result  # 결과를 반환
