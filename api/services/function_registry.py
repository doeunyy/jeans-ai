from api.functions.friend.delete_friend import process_delete_friend
from api.functions.friend.send_friend_request import process_send_friend_request
from api.functions.friend.accept_friend_request import process_accept_friend_request
from api.functions.search.search_photo import process_photo_search


# 기능별 처리 함수 매핑
FUNCTION_MAPPINGS = {
    ### 친구
    "send_friend_request": process_send_friend_request,
    "accept_friend_request": process_accept_friend_request,
    "delete_friend": process_delete_friend,
    
    ### 검색
    "search_photo": process_photo_search,
    
    ### 설정
    
    # 추가 기능이 있다면 여기에 추가
}

# 기능별 front-end path 매핑
FRONT_PATHS = {
    ### 홈
    "home": "/Home/main-page",
    
    ### 검색
    "search_photo": "/Home/search",
    
    ### 공유
    "share": "/Share/share-select-img",
    
    ### 친구
    "view_friends_list": "/Friend/list",
    "send_friend_request": "/Friend/send-search",
    "accept_friend_request": "/Friend/receive",
    "delete_friend": "/Friend/delete",
    
    ### 설정
    "edit_user_info": "/MyPage/edit-info",
    "change_profile_photo": "/MyPage/edit-info",
    "change_name": "/MyPage/edit-info",
    "change_password": "/ChangePassword/origin-password",
    "adjust_correction_value": "/Set/photo-selection0",
    "adjust_font_size": "/MyPage/word-size",
    "view_manual": "/MyPage/manual",
    
    ### 로그아웃/탈퇴
    "logout": "/explore",
    "quit": "/MyPage/quit",
}


def get_function_handler(function_name: str):
    """
    기능 이름을 받아 적절한 처리 함수를 반환.
    """
    return FUNCTION_MAPPINGS.get(function_name)

def get_front_path(function_name: str):
    """
    기능 이름을 받아 해당 front-end path의 key와 value를 반환.
    """
    return (function_name, FRONT_PATHS.get(function_name, ""))  # 기본값은 빈 문자열