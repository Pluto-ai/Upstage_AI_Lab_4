import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback

import wandb

# peft 관련 모듈 임포트
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# config 설정에 tokenizer 모듈이 사용되므로 미리 tokenizer를 정의해줍니다.
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base")

# Config 설정에 필요한 토큰을 업데이트합니다.
config_data = {
    "general": {
        "data_path": "../data/",
        "model_name": "KETI-AIR/ke-t5-small",
        "output_dir": "./"
    },
    "tokenizer": {
        "encoder_max_len": 512,
        "decoder_max_len": 100,
        "bos_token": "",
        "eos_token": tokenizer.eos_token,
        "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    },
    "training": {
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "learning_rate": 5e-4,  # 학습률 증가
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": 'cosine',
        "optim": 'paged_adamw_8bit',  # 옵티마이저 변경
        "gradient_accumulation_steps": 2,
        "evaluation_strategy": 'epoch',
        "save_strategy": 'epoch',
        "save_total_limit": 5,
        "fp16": True,
        "load_best_model_at_end": True,
        "seed": 42,
        "logging_dir": "./logs",
        "logging_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": 100,
        "do_train": True,
        "do_eval": True,
        "early_stopping_patience": 5,
        "early_stopping_threshold": 0.001,
        "report_to": "wandb",
        "gradient_checkpointing": True  # Gradient Checkpointing 사용
    },
    "wandb": {
        "entity": "kd100150-chonnam-national-university",
        "project": "Upstage-NLP",
        "name": "base_line_t5_model",
    },
    "inference": {
        "ckt_path": "./best_model",
        "result_path": "./prediction/",
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "generate_max_length": 100,
        "generate_min_length": 10,
        "num_beams": 4,
        "batch_size": 8,
        "remove_tokens": [tokenizer.pad_token]
    },
    "generation": {
        "min_length": 10
    }
}

# 모델의 구성 정보를 YAML 파일로 저장합니다.
config_path = "./config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config_data, file, allow_unicode=True)
    
# 저장된 config 파일을 불러옵니다.
with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)
    
# wandb 설정 업데이트
loaded_config['wandb']['entity'] = "kd100150-chonnam-national-university"
loaded_config['wandb']['name'] = "base_line_t5_model"
loaded_config['wandb']['project'] = "Upstage-NLP"

# config에 저장된 데이터 경로를 통해 train과 validation data를 불러옵니다.
data_path = loaded_config['general']['data_path']

# train data의 구조와 내용을 확인합니다.
train_df = pd.read_csv(os.path.join(data_path,'train.csv'))

# validation data의 구조와 내용을 확인합니다.
val_df = pd.read_csv(os.path.join(data_path,'dev.csv'))

## 확인해보니 ㅋㅋ 가 포함되어 있는 데이터 존재. 이를 웃기다 정도로 대체
# replace 를 사용하여 ㅋㅋ 를 웃기다로 대체 후 저장 
train_df['dialogue'] = train_df['dialogue'].apply(lambda x:x.replace('ㅋㅋ', '웃기다'))

# 데이터 전처리를 위한 클래스 정의
class Preprocess:
    def __init__(self,
                 bos_token: str,
                 eos_token: str,
                 task_prefix: str = "summarize: "
                 ) -> None:

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.task_prefix = task_prefix

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname', 'dialogue', 'summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname', 'dialogue']]
            return test_df

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue'].apply(lambda x: self.task_prefix + x)
            return encoder_input.tolist()
        else:
            encoder_input = dataset['dialogue'].apply(lambda x: self.task_prefix + x)
            decoder_output = dataset['summary'].apply(lambda x: x + self.eos_token)
            return encoder_input.tolist(), decoder_output.tolist()

# Dataset 클래스 정의
class DatasetForTrain(Dataset):
    def __init__(self, tokenizer, encoder_inputs, labels, config):
        self.encoder_inputs = encoder_inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        input_encodings = self.tokenizer(
            self.encoder_inputs[idx],
            max_length=self.config['tokenizer']['encoder_max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encodings = self.tokenizer(
            self.labels[idx],
            max_length=self.config['tokenizer']['decoder_max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_encodings.input_ids.squeeze()
        labels = labels.to(torch.long)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encodings.input_ids.squeeze(),
            'attention_mask': input_encodings.attention_mask.squeeze(),
            'labels': labels
        }

class DatasetForVal(DatasetForTrain):
    pass

class DatasetForInference(Dataset):
    def __init__(self, tokenizer, encoder_inputs, ids, config):
        self.encoder_inputs = encoder_inputs
        self.ids = ids
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        input_encodings = self.tokenizer(
            self.encoder_inputs[idx],
            max_length=self.config['tokenizer']['encoder_max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encodings.input_ids.squeeze().to(torch.long),
            'attention_mask': input_encodings.attention_mask.squeeze().to(torch.long),
            'ID': self.ids[idx]
        }

# 데이터셋 준비 함수
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path, 'train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    # 데이터프레임 구축
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-' * 150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-' * 150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

    encoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-' * 10, 'Load data complete', '-' * 10)

    # 데이터셋 생성
    train_dataset = DatasetForTrain(tokenizer, encoder_input_train, decoder_output_train, config)
    val_dataset = DatasetForVal(tokenizer, encoder_input_val, decoder_output_val, config)

    print('-' * 10, 'Make dataset complete', '-' * 10)
    return train_dataset, val_dataset

def compute_metrics(pred):
    rouge = Rouge()

    # pred.predictions가 tuple인 경우 대비
    if isinstance(pred.predictions, tuple):
        predictions = pred.predictions[0]
    else:
        predictions = pred.predictions

    # 만약 predictions가 logits일 경우 argmax로 토큰 ID로 변환
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    labels = pred.label_ids

    # Convert tensors to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 레이블의 -100을 pad_token_id로 변경
    labels[labels == -100] = tokenizer.pad_token_id

    # 토큰 ID를 텍스트로 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 불필요한 공백 제거
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # 디코딩 결과 출력 (필요에 따라 생략 가능)
    print('-' * 150)
    print(f"PRED: {decoded_preds[0]}")
    print(f"GOLD: {decoded_labels[0]}")
    print('-' * 150)
    print(f"PRED: {decoded_preds[1]}")
    print(f"GOLD: {decoded_labels[1]}")
    print('-' * 150)
    print(f"PRED: {decoded_preds[2]}")
    print(f"GOLD: {decoded_labels[2]}")

    # 예측 또는 레이블이 빈 문자열인 경우 처리
    if any(len(pred) == 0 for pred in decoded_preds) or any(len(label) == 0 for label in decoded_labels):
        print("Warning: Empty predictions or labels detected.")
        return {'rouge-l': 0.0}

    # ROUGE 점수 계산
    results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)

    # F-1 score를 통해 평가
    result = {key: value["f"] * 100 for key, value in results.items()}
    return result

# Trainer 로드 함수 수정
def load_trainer_for_train(config, model, tokenizer, train_dataset, val_dataset):
    print('-' * 10, 'Make training arguments', '-' * 10)
    # TrainingArguments 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to'],
        gradient_checkpointing=config['training']['gradient_checkpointing']
    )

    # wandb 초기화
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
    )

    # 모델 체크포인트를 wandb에 저장하도록 환경 변수 설정
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "false"

    # EarlyStopping 콜백 설정
    my_callback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    print('-' * 10, 'Make training arguments complete', '-' * 10)
    print('-' * 10, 'Make trainer', '-' * 10)

    # Trainer 클래스 정의
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[my_callback]
    )
    print('-' * 10, 'Make trainer complete', '-' * 10)

    return trainer

# 모델과 토크나이저 로드 함수 수정
def load_tokenizer_and_model_for_train(config, device):
    print('-' * 10, 'Load tokenizer & model', '-' * 10)
    print('-' * 10, f'Model Name : {config["general"]["model_name"]}', '-' * 10)
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 스페셜 토큰 추가
    tokenizer.add_tokens(config['tokenizer']['special_tokens'])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # 8-bit 양자화 적용
        device_map='auto'
    )
    model.resize_token_embeddings(len(tokenizer))

    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 모델을 8-bit 훈련에 적합하도록 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["layers.*.SelfAttention.o"],  # 작은 모델에 맞게 수정
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 출력
    model.print_trainable_parameters()

    # 모델 설정 수정
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.min_length = config['generation']['min_length']

    # Gradient Checkpointing 활성화
    model.gradient_checkpointing_enable()

    print(model.config)

    # 토큰이 올바르게 추가되었는지 확인
    for tok in config['tokenizer']['special_tokens']:
        print(f"Token: {tok}, Token ID: {tokenizer.convert_tokens_to_ids(tok)}")

    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)
    return model, tokenizer

# 메인 함수 정의
def main(config):
    # device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('-' * 10, f'device : {device}', '-' * 10)
    print(torch.__version__)

    # 스페셜 토큰 추출 및 config 업데이트
    existing_special_tokens = config['tokenizer']['special_tokens']

    def reg_masking(text):
        pattern = r"#\w+#"
        return re.findall(pattern, text)

    def reg_person(text):
        pattern = r"#\w+\d+#"
        return re.findall(pattern, text)

    # 모든 대화문 합치기
    all_dialogues = pd.concat([train_df['dialogue'], val_df['dialogue']])

    # 스페셜 토큰 집합 생성
    unique_tokens = set()

    for text in all_dialogues:
        tokens = reg_masking(text)
        unique_tokens.update(tokens)

        person_tokens = reg_person(text)
        unique_tokens.update(person_tokens)

    # 합친 스페셜 토큰 리스트로 config 업데이트
    all_special_tokens = list(set(existing_special_tokens + list(unique_tokens)))

    # 합친 스페셜 토큰 리스트로 config 업데이트
    config['tokenizer']['special_tokens'] = all_special_tokens

    # 모델과 토크나이저 로드
    model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    print('-' * 10, "tokenizer special tokens : ", tokenizer.special_tokens_map, '-' * 10)

    # 데이터셋 준비
    preprocessor = Preprocess(bos_token="", eos_token=config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_dataset, val_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    # Trainer 로드
    trainer = load_trainer_for_train(config, model, tokenizer, train_dataset, val_dataset)
    trainer.train()  # 모델 학습 시작
    config['inference']['ckt_path'] = trainer.state.best_model_checkpoint

    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main(loaded_config)

# 추론을 위한 데이터셋 준비 함수
def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    print('-' * 150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-' * 150)

    encoder_input_test = preprocessor.make_input(test_data, is_test=True)
    print('-' * 10, 'Load data complete', '-' * 10)

    test_dataset = DatasetForInference(tokenizer, encoder_input_test, test_id.tolist(), config)
    print('-' * 10, 'Make dataset complete', '-' * 10)
    
    return test_data, test_dataset

# 추론을 위한 모델과 토크나이저 로드 함수 수정
def load_tokenizer_and_model_for_test(config, device):
    print('-' * 10, 'Load tokenizer & model for inference', '-' * 10)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-' * 10, f'Model Name : {model_name}', '-' * 10)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(config['tokenizer']['special_tokens'])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map='auto'
    )
    model.resize_token_embeddings(len(tokenizer))

    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 모델을 8-bit 훈련에 적합하도록 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정 (추론 시에도 필요)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v", "k", "o", "wi", "wo"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)

    # 학습된 LoRA 가중치 로드
    model.load_state_dict(torch.load(os.path.join(ckt_path, "pytorch_model.bin")), strict=False)

    model.to(device)

    # 토큰이 올바르게 추가되었는지 확인
    for tok in config['tokenizer']['special_tokens']:
        print(f"Token: {tok}, Token ID: {tokenizer.convert_tokens_to_ids(tok)}")

    print('-' * 10, 'Load tokenizer & model complete', '-' * 10)

    return model, tokenizer

# 추론 함수
def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('-' * 10, f'device : {device}', '-' * 10)
    print(torch.__version__)

    model, tokenizer = load_tokenizer_and_model_for_test(config, device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                min_length=config['inference']['generate_min_length'],  # 추가된 부분
                num_beams=config['inference']['num_beams'],
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            summary.extend(decoded)

    # 불필요한 토큰 제거
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, "").strip() for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary": preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output

# 학습된 모델의 test를 진행합니다.
if __name__ == "__main__":
    output = inference(loaded_config)

output
