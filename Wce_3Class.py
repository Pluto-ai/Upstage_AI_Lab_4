import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
from tqdm import tqdm
import wandb
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

# 시드 고정
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# W&B Sweep 설정
sweep_config = {
    "method": "random",
    "metric": {"name": "val_macro_f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [1e-3]},
        "batch_size": {"values": [16]},
        "epochs": {"values": [30]},
        "optimizer": {"values": ["adam"]}
    }
}

# W&B Sweep 초기화
sweep_id = wandb.sweep(sweep_config, project="V4-efficientnet")

# 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None, extra_transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(labels_file)
        self.transform = transform
        self.extra_transform = extra_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.iloc[idx, 1])
        
        # 특정 클래스 (3, 4)에 대해 추가 증강 적용
        if label in [3, 4] and self.extra_transform:
            image = self.extra_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 데이터 전처리 변환 설정
img_size = 384
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 추가 증강 설정 (3, 4번 클래스에만 적용, 확률적으로 적용)
extra_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
    ], p=0.3)  # 30% 확률로 증강 적용
])


# 모델 저장 경로 설정
save_dir = '/root/data/home/code/model/save_dir/V7_EfficientNet'
os.makedirs(save_dir, exist_ok=True)

# train_and_evaluate 함수 정의
def train_and_evaluate():
    # W&B 초기화
    wandb.init(project="V7-efficientnet", group="EfficientNet-DocumentClassification", job_type="training")
    config = wandb.config

    # Early Stopping 설정
    patience = 5
    min_delta = 0.001
    best_val_loss = float('inf')
    no_improvement_count = 0

    # 데이터셋 로드 및 학습/검증 세트 분할
    dataset = CustomImageDataset(
        img_dir='/root/data/home/data/V7_train', 
        labels_file='/root/data/home/data/V7_train_labels.csv', 
        transform=transform,
        extra_transform=extra_transform  # 추가 증강 적용
    )
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.df.iloc[:, 1]
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    # EfficientNet-B4 모델 로드 및 수정
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, 17)  # 클래스 개수에 맞춰 수정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 옵티마이저 선택
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Cosine Annealing 스케줄러 추가
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 클래스 가중치를 계산하여 손실 함수 정의
    class_counts = dataset.df.iloc[:, 1].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 학습 함수 정의
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        return running_loss / len(train_loader), accuracy, macro_f1

    # 평가 함수 정의
    def evaluate(model, val_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        return running_loss / len(val_loader), accuracy, macro_f1

    # 학습 및 평가 루프
    for epoch in range(config.epochs):
        torch.cuda.empty_cache()  # 매 에포크 이후 메모리 정리
        train_loss, train_accuracy, train_macro_f1 = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_macro_f1 = evaluate(model, val_loader, criterion, device)

        # wandb에 로그 기록
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        print(f"Epoch [{epoch+1}/{config.epochs}], train_loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train Macro F1: {train_macro_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val Macro F1: {val_macro_f1:.4f}")

        # 스케줄러 스텝
        scheduler.step()

        # Early stopping 조건 체크
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improvement_count = 0
            # 베스트 모델 저장
            model_path = os.path.join(save_dir, f'best_model_{wandb.run.name or wandb.run.id}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved to {model_path}')
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
    wandb.finish()

# Sweep 에이전트 실행 (한 번만 학습)
wandb.agent(sweep_id, function=train_and_evaluate, count=1)