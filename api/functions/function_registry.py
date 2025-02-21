import json
import os


def get_function_list() -> list:
    functions = []
    base_path = "api/functions"
    
    # functions 디렉토리 내의 모든 .json 파일을 찾아서 로드
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        functions.append(json.load(f))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {file_path}: {e}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return functions