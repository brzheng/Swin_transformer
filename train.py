# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from utils.dataset import ClassificationDataset
from models.swin_classifier import SwinPureClassifier

class EarlyStopping:
    def __init__(self, patience=8, delta=0.0):
        """
        Args:
            patience (int): 忍耐轮数。如果连续这么多轮 Val Acc 都没有提升，就触发早停。
            delta (float): 算作提升的最小变化量。通常设为 0。
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False

    def __call__(self, val_acc):
        # 如果当前的验证集准确率比历史最高还要高（超出 delta 阈值）
        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter = 0
            return True       # 返回 True 提示 train.py 需要保存模型权重
        else:
            self.counter += 1 # 关键：没破纪录，计数器累加一轮
            print(f"    早停提示: Val Acc 没有提升。持续轮数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True # 触发早停信号
            return False

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """训练一个 Epoch 的函数"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 渲染精美的训练进度条
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)    # 梯度裁剪
        optimizer.step()

        # 统计损失与准确率
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100.0 * correct / total:.2f}%"
        })

    # 更新学习率调度器
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证集/测试集评估函数"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    return val_loss, val_acc


def main():
    print("正在读取数据集...")
    full_dataset = ClassificationDataset(img_dir=config.TRAIN_DIR, mode='val')
    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError(f"错误：在 {config.TRAIN_DIR} 路径下没有找到任何数据，请检查 config.py 配置！")
    # 80% for train and 20% for test
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_set.dataset.transform = ClassificationDataset(img_dir=config.TRAIN_DIR, mode='train').transform

    # 4. 构建 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"数据加载完成！总样本数: {total_size}")
    print(f" ├─ 训练集 (80%): {len(train_set)} 样本")
    print(f" └─ 验证集 (20%): {len(val_set)} 样本")
    print(f"任务分类总数: {config.NUM_CLASSES}")

    # 初始化纯分类 Swin 模型
    model = SwinPureClassifier(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    print(f"使用模型：{model.__class__.__name__}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=8, delta=0.0)

    # 核心训练与验证循环
    best_acc = 0.0
    print(f"开始训练，当前设备: {config.DEVICE}")

    for epoch in range(1, config.EPOCHS + 1):
        print(f"Epoch {epoch}/{config.EPOCHS} ---")
        print(f"当前 (LR): {optimizer.param_groups[0]['lr']:.6f}")

        # 训练一个 Epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, config.DEVICE
        )

        # 验证当前模型性能
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}   |   Val Acc: {val_acc * 100:.2f}%")

        # 使用早停机制来决定是否保存模型以及是否中断整个训练
        is_best = early_stopping(val_acc)
        # 保存表现最好的模型
        if is_best:
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"=> Saved best model with Val Acc: {val_acc:.4f}")

        if early_stopping.early_stop:
            print(f"触发早停机制！模型连续 {early_stopping.patience} 轮未提升，已强制中断训练以保护泛化性。")
            break

    print(f"训练完成!")
    print(f"💾 最优模型权重已保存至: {config.SAVE_PATH}")


if __name__ == "__main__":
    main()