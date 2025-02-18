'''
Fine-tuning whisper (possible: tiny, small, medium, etc.)

References:
    - Master reference:

Written by: Doeun Kim
Licence: MIT
'''

import argparse
from datasets import load_dataset, DatasetDict

def get_config() -> argparse.ArgumentParser:
    '''Whisper finetuning args parsing functoin'''
    parser = argparse.ArgumentParser()
    
    

class Trainer:

    def __init__(self, config) -> None:
        '''Init all required args for whisper finetune'''
        pass


    def load_dataset(self, ) -> DatasetDict:
        '''Load dataset containing tain/valid/test'''
        dataset = DatasetDict()
        dataset['train'] = load_dataset(
            path='csv', 
            split='train',
            data_files="./dataset/dataset_train_updated.csv")
        pass 

    def compute_metrics(self, pred) -> dict:
        '''Prepare evaluation metric (wer, cer, etc.)'''
        pass

    def prepare_dataset(self, batch):
        '''Get input features with numpy array & sentence label'''
        pass

    def process_dataset(self, dataset) -> tuple:
        '''Process loaded dataset applying prepare_dataset)'''
        pass

    def enforce_finetune_lang(self) -> None:
        '''Enforce finetuning language'''
        pass

    def create_trainer(self, train, valid) -> None:
        '''Create seq2seq trainer'''
        pass

    def run(self) -> None:
        '''Run trainer'''
        pass

if __name__ == '__main__':
    config = get_config()
    trainer = Trainer(config)
    trainer.run()