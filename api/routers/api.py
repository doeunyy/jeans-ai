from fastapi import APIRouter
from pydantic import BaseModel
from api.services import gpt_service
# from api.schemas import TextInput
# from api.utils import get_function_list
from api.functions.function_registry import get_function_list

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/delete-friend")
async def process_text(input_data: TextInput):
    """
    친구 삭제하는 하는 API 엔드포인트
    """
    functions = get_function_list()
    response = await gpt_service.process_with_functions(input_data.text, functions)
    return response


@router.post("/search-photo")
async def search_photo_endpoint(input_data: TextInput):
    functions = get_function_list()
    search_result = await gpt_service.process_with_functions(input_data.text, functions)
    return search_result


@router.post("/accept-friend-request")
async def accept_friend_request_endpoint(input_data: TextInput):
    """
    친구 요청을 수락하는 API 엔드포인트
    """
    functions = get_function_list()
    accept_result = await gpt_service.process_with_functions(input_data.text, functions)
    return accept_result


