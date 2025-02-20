'''
Fine-tuning whisper (possible: tiny, small, medium, etc.)

References:
    - Master reference: https://huggingface.co/blog/fine-tune-whisper

Written by: Doeun Kim
Licence: MIT
'''

import argparse
import numpy as np
import os
import evaluate
from datasets import load_dataset, DatasetDict, Dataset
from trainer.collator import DataCollatorSpeechSeq2SeqWithPadding
from utils import get_unique_directory
from transformers import ( 
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
) 
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from pprint import pprint

def get_config():
    '''Whisper finetuning args parsing functoin'''
    parser = argparse.ArgumentParser()
    
    ## dataset
    parser.add_argument(
        '--train-set', '-t',
        required=True,
        help='Training dataset name (file name or file path)'
    )
    parser.add_argument(
        '--valid-set', '-v',
        required=True,
        help='Validation dataset name (file name or file path)'
    )
    parser.add_argument(
        '--test-set', '-e',
        # required=True,
        default='C:/Users/004/Desktop/Jeans/AI/whisper/dataset/dataset_test.csv',
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
        
        # Model
        self.model = WhisperForConditionalGeneration.from_pretrained(self.pretrained_model)
        
        # Label collator
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # Training args 
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,     # change to a repo name of your choice
            per_device_train_batch_size=16, # GPU 성능에 다라 16 -> 32 변경 가능
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,     
            warmup_steps=500,               # gradient exploding 방지를 위한 warm-up 과정
            # max_steps=5000,
            gradient_checkpointing=True,
            fp16=True,                      # 부동 소수점 자리 수 (default: fp32 -> fp16 - speed-up training)
            eval_strategy="steps",    
            per_device_eval_batch_size=8,   # GPU 성능에 따라 8 -> 16 -> 32 변경 가능
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=100,               # 25 -> 100으로 변경함
            # report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model=config.metric,
            greater_is_better=False,
            push_to_hub=False,
        )


    def load_dataset(self, ) -> DatasetDict:
        '''Load dataset containing tain/valid/test'''
        dataset = DatasetDict()

        # 상대 경로 -> 절대 경로로 변환
        train_path = os.path.abspath(self.config.train_set)
        valid_path = os.path.abspath(self.config.valid_set)
        test_path = os.path.abspath(self.config.test_set)

        print(f"Resolved Train Path: {train_path}")
        print(f"Resolved Valid Path: {valid_path}")
        print(f"Resolved Test Path: {test_path}")
        
        dataset['train'] = load_dataset(
            path='csv',
            split='train',
            data_files=train_path)
        dataset['valid'] = load_dataset(
            path='csv', 
            split='train',
            data_files=valid_path)
        dataset['test'] = load_dataset(
            path='csv', 
            split='train',
            data_files=test_path)
        
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
        
        print('\nStart train dataset mapping...')
        print(dataset['train'])
        train = dataset['train'].map(
            function=self.prepare_dataset,
            remove_columns=dataset.column_names['train'],
            num_proc=4
        )
        
        print('\nStart valid dataset mapping...')
        print(dataset['valid'])
        valid = dataset['valid'].map(
            function=self.prepare_dataset,
            remove_columns=dataset.column_names['valid'],
            num_proc=4
        )
        
        print('\nStart test dataset mapping...')
        print(dataset['test'])
        test = dataset['test'].map(
            function=self.prepare_dataset,
            remove_columns=dataset.column_names['test'],
            num_proc=4
        )
        
        return (train, valid, test)

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
        trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=train,
            eval_dataset=valid,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        return trainer

    def run(self) -> None:
        '''Run trainer'''
        self.enforce_finetune_lang()
        dataset = self.load_dataset()
        train, valid, test = self.process_dataset(dataset)
        trainer = self.create_trainer(train, valid)
        
        # 모델 학습 시작
        print('\nStart fine-tuning...')
        trainer.train()
        trainer.save_model(self.finetuned_model_dir)
        
        # 모델 성능 평가
        print('\nStart testing performance using test_dataset...')
        result_dic = trainer.evaluate(eval_dataset=test)
        pprint(result_dic)
        
        print('\nClearing GPU cache')
        torch.cuda.empty_cache()
        print('\nFine-tuning is done!')

if __name__ == '__main__':
    config = get_config()
    trainer = Trainer(config=config)
    # dataset = trainer.load_dataset()
    # print(dataset)
    # train, valid, test = trainer.process_dataset(dataset)
    
    trainer.run()
    
    # dataset = trainer.load_dataset()
    
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
    # train, valid = trainer.process_dataset(dataset)