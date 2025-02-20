'''
Utilities for fine-tuning
'''

from datetime import datetime
import pickle
from random import shuffle
import pandas as pd
import numpy as np
import librosa as lr
import soundfile as sf
import os
from tqdm import tqdm


def get_unique_directory(dir_name: str, model_name: str) -> str:
    '''입력된 디렉토리 이름에 날짜/시간 정보를 추가해서 '''
    model_name = model_name.split('/')[-1]
    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    return os.path.join(dir_name, f'{model_name}-{now}')