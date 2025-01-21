"""
配置模块，用于管理模型训练和评估的所有参数设置

主要功能：
1. 提供默认配置参数
2. 支持从命令行参数更新配置
3. 管理模型、训练和数据相关的所有参数
"""

import torch

class Config:
    """
    配置类，包含所有模型训练和评估需要的参数
    
    配置参数分类：
    1. 训练相关参数（批次大小、学习率、轮次等）
    2. 模型相关参数（预训练模型、特征维度等）
    3. 数据相关参数（数据目录、文件路径等）
    4. 实验控制参数（是否使用wandb、交叉验证等）
    5. 损失函数相关参数（损失类型、权重等）
    """
    
    def __init__(self, args=None):
        """
        初始化配置对象
        
        参数:
            args: 命令行参数对象，包含用户定义的配置值
                 若为None则使用默认配置
        
        配置项说明：
        1. 基础训练参数：
            - batch_size: 训练批次大小
            - learning_rate: 学习率
            - num_epochs: 训练轮次
            - val_ratio: 验证集比例
            - dropout: dropout比率
            
        2. 模型相关参数：
            - text_model_name: 文本预训练模型路径
            - image_model_name: 图像预训练模型路径
            - text_dim: 文本特征维度
            - image_dim: 图像特征维度
            - feature_fusion: 特征融合方式
            
        3. 数据相关参数：
            - data_dir: 数据目录
            - train_file: 训练文件名
            - test_file: 测试文件名
            - result_file: 结果文件名
            
        4. 实验控制参数：
            - wandb: 是否使用wandb记录实验
            - early_stop_patience: 早停耐心值
            - use_kfold: 是否使用交叉验证
            - k_folds: 交叉验证折数
            - project_name: 项目名称
            - log_iteration: 日志记录间隔
            
        5. 模态控制参数：
            - use_text: 是否使用文本模态
            - use_image: 是否使用图像模态
            
        6. 损失函数参数：
            - loss_type: 损失函数类型
            - alpha: 损失权重alpha
            - beta: 损失权重beta
            - neural_init_weight: neutral类初始权重
        """
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从命令行参数更新配置
        if args:
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.num_epochs = args.num_epochs
            self.val_ratio = args.val_ratio
            self.wandb = args.wandb
            self.early_stop_patience = args.early_stop_patience
            self.text_model_name = args.text_model_name
            self.image_model_name = args.image_model_name
            self.data_dir = args.data_dir
            self.train_file = args.train_file
            self.test_file = args.test_file
            self.result_file = args.result_file
            self.use_kfold = args.use_kfold
            self.k_folds = args.k_folds
            self.project_name = args.project_name
            self.use_text = args.use_text
            self.use_image = args.use_image
            self.feature_fusion = args.feature_fusion
            self.num_classes = args.num_classes
            self.log_iteration = args.log_iteration
            self.name = args.name
            self.text_dim = args.text_dim
            self.image_dim = args.image_dim
            self.dropout = args.dropout
            self.loss_type = args.loss_type
            self.alpha = args.alpha
            self.beta = args.beta
            self.neural_init_weight = args.neural_init_weight
        else:
            # 默认配置
            self.batch_size = 32
            self.learning_rate = 1e-4
            self.num_epochs = 10
            self.val_ratio = 0.2
            self.wandb = False
            self.early_stop_patience = 3
            self.text_model_name = "./pretrained_models/bert-base-uncased"
            self.image_model_name = "./pretrained_models/swinv2-base"
            self.data_dir = "data"
            self.train_file = "train.txt"
            self.test_file = "test_without_label.txt"
            self.result_file = "result.txt"
            self.use_kfold = False
            self.k_folds = 5
            self.project_name = "multimodal_sentiment_analysis"
            self.use_text = True
            self.use_image = True
            self.feature_fusion = "concat"
            self.num_classes = 3
            self.log_iteration = 10
            self.name = "default"
            self.text_dim = 256
            self.image_dim = 128
            self.dropout = 0.1
            self.loss_type = 'focal'
            self.alpha = 0.5
            self.beta = 0.5
            self.neural_init_weight = 1.0
