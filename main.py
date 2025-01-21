"""
该脚本用于训练和评估多模态情感分析模型。它支持单次训练和K折交叉验证两种模式，并提供了灵活的命令行参数配置。

主要功能包括：
1. 设置随机种子以确保实验的可重复性
2. 支持多种特征融合策略（concat、attention、combine等）
3. 支持多种损失函数（adaptive class balanced loss、cross entropy）
4. 支持K折交叉验证训练模式
5. 支持WandB日志记录
6. 提供数据增强和类别平衡机制
7. 支持模型预测和结果保存

使用方法：
1. 通过命令行参数指定训练参数、模型配置等
2. 运行脚本进行模型训练和评估

命令行参数说明：
- `--batch_size`：训练批次大小。默认值为32
- `--learning_rate`：学习率。默认值为1e-5
- `--num_epochs`：训练轮数。默认值为30
- `--val_ratio`：验证集比例。默认值为0.2
- `--wandb`：是否使用wandb进行日志记录。默认值为False
- `--early_stop_patience`：早停耐心值。默认值为3
- `--text_model_name`：文本预训练模型路径。默认为bert-base-uncased
- `--image_model_name`：图像预训练模型路径。默认为swinv2-base
- `--use_kfold`：是否使用K折交叉验证。默认值为False
- `--k_folds`：K折交叉验证的折数。默认值为5
- `--feature_fusion`：特征融合方式。可选值：concat/attention/combine等
- `--text_dim`：文本特征维度。默认值为256
- `--image_dim`：图像特征维度。默认值为128
- `--loss_type`：损失函数类型。可选值：acb/ce
- `--alpha`：focal损失的alpha参数。默认值为0.25
- `--beta`：边界损失的beta参数。默认值为0.75
- `--neural_init_weight`：neutral类的初始权重。默认值为1.5

示例命令：
python main.py --batch_size 32 --learning_rate 1e-5 --num_epochs 30 --wandb True --feature_fusion encoder --loss_type acb --alpha 0.25 --beta 0.75 --use_kfold True --k_folds 5

工具函数说明：
- set_seed：设置随机种子确保实验可重复性
- print_sampled_distribution：打印采样后的类别分布情况
- print_dataset_distribution：打印数据集的类别分布情况

训练流程：
1. 解析命令行参数并初始化配置
2. 加载预训练模型和数据处理器
3. 准备数据集和数据加载器
4. 根据配置选择单次训练或K折交叉验证
5. 执行训练并记录结果
6. 保存最佳模型和预测结果
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoImageProcessor  
from torchvision import transforms
from sklearn.metrics import accuracy_score
from utils.dataload import MultimodalDataset
from multimodel import MultimodalModel
from trainer import MultimodalTrainer
from utils.config import Config
from argparse import ArgumentParser
import random
import numpy as np
from sklearn.model_selection import KFold
import wandb
from torch.utils.data import WeightedRandomSampler

RANDOM_SEED = 47

def set_seed(seed=RANDOM_SEED):
    """设置随机种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_sampled_distribution(dataloader, dataset):
    """打印采样后的类别分布"""
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, _, labels in dataloader:
        for label in labels:
            label_counts[int(label)] += 1
    
    total = sum(label_counts.values())
    print("\n采样后类别分布:")
    for label, count in label_counts.items():
        print(f"类别 {label}: {count} ({count/total*100:.2f}%)")

def print_dataset_distribution(dataset, name="Dataset"):
    """打印数据集分布信息"""
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, _, label in dataset:
        label_counts[int(label)] += 1
    
    total = sum(label_counts.values())
    print(f"\n{name} 分布:")
    for label, count in label_counts.items():
        print(f"类别 {label}: {count} ({count/total*100:.2f}%)")

def main():
    """主函数"""
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 命令行参数解析
    parser = ArgumentParser(description='Train a Multimodal Model')
    
    # 添加命令行参数
    parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--wandb', type=bool, default=False, help='是否使用wandb进行日志记录')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--text_model_name', type=str, default="./pretrained_models/bert-base-uncased", help='文本预训练模型')
    parser.add_argument('--image_model_name', type=str, default="./pretrained_models/swinv2-base", help='图像预训练模型')
    parser.add_argument('--data_dir', type=str, default="data/text_image_pair", help='数据目录')
    parser.add_argument('--train_file', type=str, default="data/train.txt", help='训练文件名')
    parser.add_argument('--test_file', type=str, default="data/test_without_label.txt", help='测试文件名')
    parser.add_argument('--result_file', type=str, default="result.txt", help='预测结果文件名')
    parser.add_argument('--use_kfold', type=bool, default=False, help='是否使用K折交叉验证')
    parser.add_argument('--k_folds', type=int, default=5, help='K折交叉验证的折数')
    parser.add_argument('--project_name', type=str, default="multimodal_sentiment_analysis_loss", help='wandb项目名称')
    parser.add_argument('--use_text', type=int, choices=[0, 1], default=1, help='是否使用文本模态')
    parser.add_argument('--use_image', type=int, choices=[0, 1], default=1, help='是否使用图像模态')
    parser.add_argument('--feature_fusion', type=str, default='encoder', choices=['concat', 'attention', 'combine', 'attention_concat', 'attention_combine', 'encoder'], help='特征融合方式')
    parser.add_argument('--num_classes', type=int, default=3, help='分类类别数')
    parser.add_argument('--log_iteration', type=int, default=10, help='记录指标的迭代次数')
    parser.add_argument('--name', type=str, default="default", help='实验名称')
    parser.add_argument('--text_dim', type=int, default=256, help='文本维度')
    parser.add_argument('--image_dim', type=int, default=128, help='图像维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--loss_type', type=str, default='acb', 
                       choices=['acb', 'ce'], 
                       help='损失函数类型：acb-自适应类别平衡损失, ce-交叉熵损失')
    parser.add_argument('--alpha', type=float, default=0.25, help='focal损失的alpha参数')
    parser.add_argument('--beta', type=float, default=0.75, help='边界损失的beta参数')
    parser.add_argument('--neural_init_weight', type=float, default=1.5, help='neutral类的初始权重')

    # 解析命令行参数    
    args = parser.parse_args()
    
    # 将整数转换为布尔值
    args.use_text = True if args.use_text == 1 else False
    args.use_image = True if args.use_image == 1 else False
    
    # 初始化配置
    config = Config(args)
    
    # 打印配置信息
    print("Config Info:")
    for attr, value in vars(config).items():
        print(f"{attr}: {value}")
    
    # 初始化tokenizer和image_processor
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    
    # 修改transform，使用image_processor的推荐设置
    transform = transforms.Compose([
        transforms.Resize(
            (image_processor.size["height"], image_processor.size["width"]),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean, 
            std=image_processor.image_std
        )
    ])
    
    # 加载数据集
    full_train_dataset = MultimodalDataset(
        config.data_dir, 
        config.train_file, 
        transform,
        is_train=True,
        augment_ratio=1.5 
    )
    full_train_dataset.print_stats()
    
    test_dataset = MultimodalDataset(
        config.data_dir, 
        config.test_file, 
        transform,
        is_train=False,
        augment_ratio=1
    )
    
    # 获取原始数据索引（不包含增强数据）
    original_indices = [i for i, guid in enumerate(full_train_dataset.guid_list) if "_aug" not in guid]
    
    if config.use_kfold:
        # K折交叉验证
        kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=RANDOM_SEED)
        best_val_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(original_indices)):
            print(f"Fold {fold+1}/{config.k_folds}")
            
            # 将原始索引映射回完整数据集
            original_train_idx = [original_indices[i] for i in train_idx]
            original_val_idx = [original_indices[i] for i in val_idx]
            
            # 获取验证集的所有原始GUID
            val_guids = set([full_train_dataset.guid_list[i] for i in original_val_idx])
            
            # 获取所有增强数据的索引，并过滤掉验证集对应的增强数据
            augmented_indices = [
                i for i in range(len(full_train_dataset)) 
                if i not in original_indices and 
                full_train_dataset.guid_list[i].split("_aug")[0] not in val_guids
            ]
            
            # 训练集包含原始训练数据和过滤后的增强数据
            train_indices = original_train_idx + augmented_indices
            
            # 准备数据加载器
            train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(full_train_dataset, original_val_idx)
            
            train_dataloader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=True
            )
            
            val_dataloader = DataLoader(
                val_subset,
                batch_size=config.batch_size,
                shuffle=False
            )
            
            # 初始化模型和训练器
            model = MultimodalModel(config).to(config.device)
            trainer = MultimodalTrainer(model, tokenizer, config)
            
            # 训练模型
            best_val_acc = trainer.train(train_dataloader, val_dataloader, fold=fold)
            best_val_accs.append(best_val_acc)
            
        # 输出平均结果
        avg_val_acc = np.mean(best_val_accs)
        print(f'Average Validation Accuracy: {avg_val_acc:.4f}')
        
    else:
        # 计算划分索引
        dataset_size = len(original_indices)
        val_size = int(dataset_size * config.val_ratio)
        train_size = dataset_size - val_size
        
        # 随机划分原始数据
        train_original_idx, val_original_idx = torch.utils.data.random_split(
            original_indices, 
            [train_size, val_size]
        )
        
        # 获取验证集的所有原始GUID
        val_guids = set([full_train_dataset.guid_list[i] for i in val_original_idx.indices])
        
        # 获取所有增强数据的索引，并过滤掉验证集对应的增强数据
        augmented_indices = [
            i for i in range(len(full_train_dataset)) 
            if i not in original_indices and 
            full_train_dataset.guid_list[i].split("_aug")[0] not in val_guids
        ]
        
        # 训练集包含原始训练数据和过滤后的增强数据
        train_indices = train_original_idx.indices + augmented_indices
        
        # 创建数据加载器
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_original_idx.indices)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # 初始化模型
        model = MultimodalModel(config).to(config.device)
        
        # 初始化训练器
        trainer = MultimodalTrainer(model, tokenizer, config)
        
        # 训练模型
        best_val_acc = trainer.train(train_dataloader, val_dataloader)
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # 打印数据集分布
        # print_dataset_distribution(train_dataset, "训练集")
        # print_dataset_distribution(val_dataset, "验证集")
        
        # 检查验证集样本
        # print("\n验证集样本示例:")
        # with open("val_message.txt", "w", encoding="utf-8") as f:
        #     f.write("验证集样本示例:\n")
        #     for i in range(min(20000, len(val_dataset))):  
        #         text, image, label = val_dataset[i]
        #         sample_info = f"""
        #                         样本 {i+1}:
        #                         GUID: {val_dataset.dataset.guid_list[val_dataset.indices[i]]}
        #                         文本: {text[:50]}...
        #                         标签: {label}
        #                         图像大小: {image.size()}
        #                         {"-" * 40}
        #                         """
        #         print(sample_info)
        #         f.write(sample_info)
                
        # 打印训练集样本示例
        # with open("train_message.txt", "w", encoding="utf-8") as f:
        #     f.write("训练集样本示例:\n")
        #     for i in range(min(20000, len(train_dataset))):  
        #         text, image, label = train_dataset[i]
        #         sample_info = f"""
        #                         样本 {i+1}:
        #                         GUID: {train_dataset.dataset.guid_list[train_dataset.indices[i]]}
        #                         文本: {text[:50]}...
        #                         标签: {label}
        #                         图像大小: {image.size()}
        #                         {"-" * 40}
        #                         """
        #         print(sample_info)
        #         f.write(sample_info)
                    

    # 预测测试集
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    predictions = trainer.predict(test_dataloader)
    
    # 将数字标签转换为文本标签
    label_map = {0: "positive", 1: "neutral", 2: "negative"}
    text_predictions = [label_map[pred] for pred in predictions]
    
    # 创建预测结果字典
    results = dict(zip(test_dataset.guid_list, text_predictions))
    
    # 将预测结果写回文件
    output_lines = ["guid,tag\n"]  # 添加标题行
    for guid, pred in results.items():
        output_lines.append(f"{guid},{pred}\n")
    
    # 保存预测结果
    with open(config.result_file, "w") as f:
        f.writelines(output_lines)
    
    print(f"预测结果已保存到 {config.result_file}")



if __name__ == "__main__":
    main()