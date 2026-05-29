# predict.py
import os
import torch
from torch.utils.data import DataLoader
import config
from utils.dataset import ClassificationDataset
from models.swin_classifier import SwinPureClassifier


def main():
    student_id = "only_for_test"  # 记得改成你的真实学号

    # 加载测试集数据
    test_dataset = ClassificationDataset(config.TEST_DIR, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if hasattr(test_dataset, 'label_to_idx'):
        idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
    else:
        idx_to_label = None

    # 加载训练好的模型权重
    model = SwinPureClassifier(num_classes=config.NUM_CLASSES)
    if not os.path.exists(config.SAVE_PATH):
        raise FileNotFoundError(f"错误：未找到权重文件 {config.SAVE_PATH}，请先运行 train.py 进行训练！")
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()

    results = []

    # 3. 开始推理
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # 获取当前图片的文件名
            img_name = img_names[0]
            # 获取模型预测的稠密索引
            pred_idx = preds.item()
            if idx_to_label is not None:
                pred_class_name = idx_to_label[pred_idx]
            else:
                pred_class_name = str(pred_idx)

            # 保存结果：(文件名, 预测分类名)
            results.append((img_name, pred_class_name))

    # 按文件名进行标准排序（确保输出文本整齐规范）
    results.sort(key=lambda x: x[0])

    output_filename = f"{student_id}.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for filename, pred_name in results:
            f.write(f"{filename}\t{pred_name}\n")

    print(f"成功生成批量识别结果文件: {output_filename}")


if __name__ == "__main__":
    main()