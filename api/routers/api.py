from fastapi import APIRouter
from pydantic import BaseModel
from api.services import gpt_service
from api.functions.function_registry import get_function_list

router = APIRouter()

class TextInput(BaseModel):
    text: str
    
@router.post("/process")
async def process_request(input_data: TextInput):
    """
    GPT를 사용하여 입력된 텍스트를 분석하고 적절한 기능을 실행하는 엔드포인트
    """
    functions = get_function_list()  # 사용 가능한 기능 목록 가져오기
    result = await gpt_service.process_with_functions(input_data.text, functions)
    return result