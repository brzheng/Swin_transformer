# eval_predictions.py
import os

def evaluate_predictions(txt_path):
    print(f"====== 开始评估结果文件: {txt_path} ======")

    if not os.path.exists(txt_path):
        print(f"错误：找不到文件 {txt_path}，请先运行 predict.py 生成它。")
        return

    total_count = 0
    correct_count = 0
    wrong_lines = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 按照制表符 \t 切分
            parts = line.split("\t")
            if len(parts) != 2:
                print(
                    f"警告：第 {line_idx} 行格式不正确（没有包含由 \\t 分隔的两个参数），已跳过。"
                )
                continue

            filename, pred_label = parts[0], parts[1]

            # 从文件名中提取真实类别号 (例如 "001_42.png" -> "001")
            # 兼容处理：如果文件名带路径，先取纯文件名
            pure_filename = os.path.basename(filename)

            if "_" in pure_filename:
                # 提取下划线前的部分作为真实分类名
                true_label = int(pure_filename.split("_")[0])
            else:
                print(
                    f"警告：第 {line_idx} 行的文件名 '{pure_filename}' 不符合 'xxx_yyy.png' 的命名规律，无法提取真实标签。"
                )
                continue

            total_count += 1

            # 去除两边可能存在的空格或特定格式后进行比对
            if str(true_label).strip() == str(pred_label).strip():
                correct_count += 1
            else:
                # 记录分类错误的详细信息
                wrong_lines.append(
                    {
                        "line": line_idx,
                        "filename": pure_filename,
                        "true": true_label,
                        "pred": pred_label,
                    }
                )

    # ====== 结果统计与输出 ======
    print("\n====== 评估结果报告 ======")
    if total_count == 0:
        print("没有找到可用于对比的有效数据行。")
        return

    accuracy = (correct_count / total_count) * 100
    print(f"总评估样本数 : {total_count} 张")
    print(f"预测正确数量 : {correct_count} 张")
    print(f"预测错误数量 : {len(wrong_lines)} 张")
    print(f"最终准确率 (Accuracy): {accuracy:.2f}%")
    print("============================\n")

    if wrong_lines:
        print("以下是预测不正确的样本明细：")
        print(f"{'行号':<6}{'文件名':<25}{'真实标签':<12}{'预测标签':<12}")
        print("-" * 60)
        for item in wrong_lines:
            print(
                f"{item['line']:<6}{item['filename']:<25}{item['true']:<12}{item['pred']:<12}"
            )
    else:
        print("所有样本预测完全正确，准确率 100%！")


if __name__ == "__main__":
    TARGET_TXT = "only_for_test.txt"

    evaluate_predictions(TARGET_TXT)