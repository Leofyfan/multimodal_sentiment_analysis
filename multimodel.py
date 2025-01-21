"""
多模态模型模块，实现了文本和图像的多模态融合模型

主要功能：
1. 支持文本和图像的特征提取
2. 提供多种特征融合策略
3. 实现多层分类器网络
4. 支持灵活的模态组合
"""

import torch
import torch.nn as nn
from transformers import BertModel, Swinv2Model


# 多模态模型类
class MultimodalModel(nn.Module):
    """
    多模态融合模型类
    
    功能：
    1. 文本特征提取（使用BERT）
    2. 图像特征提取（使用Swin Transformer V2）
    3. 多种特征融合方式：
       - concat：直接拼接
       - attention：注意力机制
       - combine：特征相加
       - attention_concat：注意力+拼接
       - attention_combine：注意力+相加
       - encoder：Transformer编码器
    4. 多层分类网络
    
    属性：
        config: 配置对象，包含模型参数
        text_model: BERT文本特征提取器
        image_model: Swin Transformer图像特征提取器
        text_proj: 文本特征投影层
        image_proj: 图像特征投影层
        transformer_encoder: 特征融合用Transformer编码器
        cross_attention: 跨模态注意力层
        classifier: 多层分类网络
    """
    def __init__(self, config):
        """
        初始化多模态模型
        
        参数:
            config: 配置对象，包含以下关键参数：
                - use_text: 是否使用文本模态
                - use_image: 是否使用图像模态
                - text_model_name: 预训练文本模型名称
                - image_model_name: 预训练图像模型名称
                - text_dim: 文本特征维度
                - image_dim: 图像特征维度
                - feature_fusion: 特征融合方式
                - dropout: dropout比率
                - num_classes: 分类类别数
        """
        super(MultimodalModel, self).__init__()
        self.config = config
        # 文本模型
        if config.use_text:
            self.text_model = BertModel.from_pretrained(config.text_model_name)
            self.text_proj = nn.Linear(768, self.config.text_dim)  # 文本特征投影
        # 图像模型
        if config.use_image:
            self.image_model = Swinv2Model.from_pretrained(config.image_model_name)
            if self.config.feature_fusion == "attention" or self.config.feature_fusion == "combine" or self.config.feature_fusion == "attention_combine" or self.config.feature_fusion == "encoder":
                self.image_proj = nn.Linear(1024, self.config.text_dim)  
            elif self.config.feature_fusion == "attention_concat":
                self.image_proj = nn.Linear(1024, self.config.image_dim)  
                self.image_proj2 = nn.Linear(self.config.image_dim, self.config.text_dim)
            else:
                self.image_proj = nn.Linear(1024, self.config.image_dim)  
            
        # 新增：Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.text_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=self.config.dropout
            ),
            num_layers=6
        )
        
        # 新增：跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.text_dim,
            num_heads=8,
            dropout=self.config.dropout
        )
        
        classifier_layers = []
        if self.config.use_text and self.config.use_image:
            if self.config.feature_fusion == "concat":
                in_features = self.config.text_dim + self.config.image_dim
            elif self.config.feature_fusion == "attention" or self.config.feature_fusion == "combine" or self.config.feature_fusion == "attention_combine" or self.config.feature_fusion == "encoder":
                in_features = self.config.text_dim
            elif self.config.feature_fusion == "attention_concat":
                in_features = self.config.text_dim * 2 + self.config.image_dim
            else:
                raise ValueError(f"Invalid feature fusion method: {self.config.feature_fusion}")
        elif self.config.use_text:
            in_features = self.config.text_dim
        else:
            in_features = self.config.image_dim
            
        # 添加多层全连接网络
        hidden_dims = [in_features, 1024, 256]  # 逐步降维
        for i in range(len(hidden_dims)-1):
            classifier_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            classifier_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))  # 批归一化
            classifier_layers.append(nn.LeakyReLU())
            classifier_layers.append(nn.Dropout(self.config.dropout))  # 较高的dropout率防止过拟合
            
        # 最后一层
        classifier_layers.append(nn.Linear(hidden_dims[-1], config.num_classes))
        
        # 使用Sequential构建分类器
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, text, image):
        """
        前向传播函数
        
        功能：
        1. 提取文本和图像特征
        2. 执行特征融合
        3. 通过分类器生成预测
        
        参数:
            text: 文本输入，包含input_ids、attention_mask等
            image: 图像输入张量
            
        返回:
            tensor: 分类预测结果
            
        特征融合流程：
        1. 分别提取文本和图像特征
        2. 根据config.feature_fusion选择融合方式：
           - concat: 直接拼接特征
           - attention: 使用跨模态注意力
           - combine: 特征相加
           - attention_concat: 注意力结果与原特征拼接
           - attention_combine: 注意力结果与原特征相加
           - encoder: 使用Transformer编码器融合
        3. 将融合特征输入分类器得到预测结果
        """
        text_features, image_features, final_features = None, None, None
        # 提取文本特征
        if self.config.use_text:
            text_outputs = self.text_model(**text)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            text_features = self.text_proj(text_features)
        # 提取图像特征
        if self.config.use_image:
            image_outputs = self.image_model(image)
            image_features = image_outputs.last_hidden_state[:, 0]  # 使用 [CLS] token
            image_features = self.image_proj(image_features)
            
        # 新增：特征融合部分
        if text_features is not None and image_features is not None:
            
            # 特征融合
            if self.config.feature_fusion == "concat":
                combined_features = torch.cat((text_features, image_features), dim=1)
                final_features = combined_features
            elif self.config.feature_fusion == "attention": 
                # 将特征堆叠
                combined_features = torch.stack([text_features, image_features], dim=0)
                # 直接使用跨模态注意力
                text_encoded = combined_features[0].unsqueeze(0)
                image_encoded = combined_features[1].unsqueeze(0)
                # 文本到图像的注意力
                attended_features, _ = self.cross_attention(
                    query=text_encoded,
                    key=image_encoded,
                    value=image_encoded
                )
                final_features = attended_features.squeeze(0)
            elif self.config.feature_fusion == "combine":
                # 直接相加融合
                final_features = text_features + image_features
            elif self.config.feature_fusion == "attention_concat":
                # 先进行concat融合
                concat_features = torch.cat((text_features, image_features), dim=1)
                # 进行attention融合
                image_features = self.image_proj2(image_features)
                combined_features = torch.stack([text_features, image_features], dim=0)
                text_encoded = combined_features[0].unsqueeze(0)
                image_encoded = combined_features[1].unsqueeze(0)
                attended_features, _ = self.cross_attention(
                    query=text_encoded,
                    key=image_encoded,
                    value=image_encoded
                )
                attention_features = attended_features.squeeze(0)
                # 将两种特征结合
                final_features = torch.cat([concat_features, attention_features], dim=1)
            elif self.config.feature_fusion == "attention_combine":
                # 先进行combine融合
                combine_features = text_features + image_features
                # 进行attention融合
                combined_features = torch.stack([text_features, image_features], dim=0)
                text_encoded = combined_features[0].unsqueeze(0)
                image_encoded = combined_features[1].unsqueeze(0)
                attended_features, _ = self.cross_attention(
                    query=text_encoded,
                    key=image_encoded,
                    value=image_encoded
                )
                attention_features = attended_features.squeeze(0)
                # 将两种特征结合
                final_features = combine_features + attention_features
            elif self.config.feature_fusion == "encoder":
                # 将文本和图像特征堆叠
                combined_features = torch.stack([text_features, image_features], dim=0)
                # 使用Transformer Encoder进行特征融合
                final_features = self.transformer_encoder(combined_features)
                # 取平均作为最终特征
                final_features = final_features.mean(dim=0)
            else:
                raise ValueError(f"Invalid feature fusion method: {self.config.feature_fusion}")
            
        elif text_features is not None:
            final_features = text_features
        else:
            final_features = image_features
            
        # 分类
        output = self.classifier(final_features)
        return output
