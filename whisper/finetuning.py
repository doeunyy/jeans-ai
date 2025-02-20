'''
Fine-tuning whisper (possible: tiny, small, medium, etc.)

References:
    - Master reference:

Written by: Doeun Kim
Licence: MIT
'''

import argparse
from datasets import load_dataset, DatasetDict
from trainer.collator import DataCollatorSpeechSeq2SeqWithPadding
from utils import get_unique_directory
from transformers import ( WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration )
from scipy.io.wavfile import read
import numpy as np
import os
import evaluate

def get_config() -> argparse.ArgumentParser:
    '''Whisper finetuning args parsing functoin'''
    parser = argparse.ArgumentParser()
    
    ## dataset
    parser.add_argument(
        '--train-set', '-t',
        # required=True,
        help='Training dataset name (file name or file path)'
    )
    parser.add_argument(
        '--valid-set', '-v',
        # required=True,
        help='Validation dataset name (file name or file path)'
    )
    
    ## model
    parser.add_argument(
        '--base-model', '-b',
        # required=True,
        default = 'openai/whisper-small',
        help='Base model for tokenizer, processor, feature_extractor. \
            ex. "openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium", etc. from huggingface'
    )
    parser.add_argument(
        '--pretrained-model', '-p',
        default='',
        help='Pre-trained model from huggingface or local. \
            If not given, we will set the model as the same as the base model'
        )
    parser.add_argument(
        '--output-dir', '-o',
        # required=True,
        default='./model_output',
        help='Output directory for the fine-tuned model'
    )
    parser.add_argument(
        '--finetuned-model-dir', '-f',
        # required=True,
        default='./model_finetuned',
        help='Directory for saving best fine-tuned model'
    )
    
    parser.add_argument(
        '--lang',
        default='ko',
        help='Language for fine-tuning (default: ko)'
    )
    parser.add_argument(
        '--task',
        default='transcribe',
        help='Task for fine-tuning (default: transcribe)'
    )
    
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=16000,
        help='Sampling rate for audio data (default: 16000)'
    )
    
    parser.add_argument(
        '--metric',
        default='cer',
        help='Evaluation metric for fine-tuning (default: cer)'
    )

    config = parser.parse_args()
    return config
    

class Trainer: 
    '''Whisper finetuning trainer'''

    def __init__(self, config) -> None:
        '''Init all required args for whisper finetune'''
        self.config = config

        # 사전 학습 모델 -> 2개
        # Base model -> tokenizer, feature_extractor, procesor
        # pretrained model -> model for fine-tuning

        # self.base_model = # 사용자 입력
        # self.pretrained_model = # 사용자 입력

        if config.pretrained_model:
            self.pretrained_model = config.pretrained_model
        else:
            print(f"\nPre-trained model not given. We will set the model as the same as the base model: {config.base_model}")
            self.pretrained_model = config.base_model
        
        self.output_dir = get_unique_directory(
            dir_name=config.output_dir,
            model_name=self.pretrained_model
            ) 
        self.finetuned_model_dir = get_unique_directory(
            dir_name=config.finetuned_model_dir,
            model_name=self.pretrained_model
            )
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Fine-tuned model directory: {self.finetuned_model_dir}\n")
        
        # Feature Extractor 
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.base_model
            )
        
        # Tokenizer
        self.tokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.base_model, 
            language=config.lang,
            task=config.task,
        )
        
        # Processor
        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path=config.base_model,
            language=config.lang,
            task=config.task,
        )
        
        # model
        self.model = WhisperForConditionalGeneration.from_pretrained(self.pretrained_model)
        
        # collator
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

    def load_dataset(self, ) -> DatasetDict:
        '''Load dataset containing tain/valid/test'''
        dataset = DatasetDict()
        
        # os에 따른 인식 오류 방지
        train_path = os.path.join(os.path.dirname(__file__), "dataset", "dataset_train.csv")
        valid_path = os.path.join(os.path.dirname(__file__), "dataset", "dataset_val.csv")
        
        dataset['train'] = load_dataset(
            path='csv', 
            # name='aihub-ko',    # (Optional) 사용자가 지정하는 이름
            split='train',
            # data_files=self.config.train_set)
            data_files=train_path)
        dataset['valid'] = load_dataset(
            path='csv', 
            # name='aihub-ko',    # (Optional) 사용자가 지정하는 이름
            split='train',
            data_files=valid_path)
        return dataset

    def compute_metrics(self, pred) -> dict:
        '''Prepare evaluation metric (wer, cer, etc.)'''
        metric = evaluate.load(self.config.metric)
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        error_rate = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {f"{self.config.metric}": error_rate}

    def prepare_dataset(self, batch):
        '''Get input features with numpy array & sentence label'''
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio_path"]
        _, data = read(audio)
        audio_array = np.array(data, dtype=np.float32)

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(
            audio_array, 
            sampling_rate=self.config.sampling_rate
            ).input_features[0] 

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["labelText"]).input_ids
        return batch

    def process_dataset(self, dataset) -> tuple:
        '''Process loaded dataset applying prepare_dataset)'''
        # common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
        train = dataset['train'].map(
            function=self.prepare_dataset,
            remove_columns=dataset['train'].column_names,
            num_proc=8
        )
        valid = dataset['valid'].map(
            function=self.prepare_dataset,
            remove_columns=dataset['valid'].column_names,
            num_proc=8
        )
        
        return (train, valid)

    def enforce_finetune_lang(self) -> None:
        '''Enforce finetuning language'''
        self.model.config.suppress_tokens = []
        self.model.generation_config.suppress_tokens = []
        self.model.config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=self.config.lang,
            task=self.config.task
        )
        self.model.generation_config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=self.config.lang,
            task=self.config.task
        )
        
    def create_trainer(self, train, valid) -> None:
        '''Create seq2seq trainer'''
        pass

    def run(self) -> None:
        '''Run trainer'''
        pass

if __name__ == '__main__':
    config = get_config()
    trainer = Trainer(config)
    # trainer.run()
    dataset = trainer.load_dataset()
    
    # # [Test code] Tokenizer 작동 확인
    # print(dataset)
    # print(dataset['train'][0])
    # print(dataset['train'][0]['labelText'])
    # input_str = dataset['train'][0]['labelText']
    # labels = trainer.tokenizer(input_str).input_ids
    # print(f"\ninput_str:\t {input_str}")
    # print(f"\nlabels:\t {labels}")
    # decoded_str_with_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=False)
    # decoded_str_without_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=True)
    # print(f"\ndecoded_str_with_special_tokens:\t {decoded_str_with_special_tokens}")
    # print(f"\ndecoded_str_without_special_tokens:\t {decoded_str_without_special_tokens}")
    # print(f'\nIs equal:\t {input_str == decoded_str_without_special_tokens}')
    
    # [Test code] Prepare & Process 
    train, valid = trainer.process_dataset(dataset)