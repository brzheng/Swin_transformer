# config.py
import torch

torch.set_num_threads(8)
# 基础配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (384, 384)
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

# 路径配置
TRAIN_DIR = "data/train"    # 训练集目录
TEST_DIR = "data/test"      # 测试集目录
SAVE_PATH = "best_model.pth"

# 类别配置
NUM_CLASSES = 100