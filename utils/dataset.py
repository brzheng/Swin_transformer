# utils/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import config

class ClassificationDataset(Dataset):
    def __init__(self, img_dir, mode='train'):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.mode = mode

        input_size = getattr(config, 'INPUT_SIZE', (384, 384))

        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                # 这两个增强对扭曲文字分类是绝对的提分核心
                transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.mode == 'test':
            return image, img_name
        else:
            label = int(img_name.split('_')[0])
            return image, label