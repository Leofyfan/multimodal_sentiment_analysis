"""
该模块实现了MultimodalTrainer类，用于多模态模型的训练、评估和预测。

主要功能：
1. 支持文本和图像的多模态训练
2. 提供模型训练、验证和预测功能
3. 支持早停机制和学习率自适应调整
4. 集成WandB日志记录
5. 支持准确率评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import wandb
import datetime
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F

class MultimodalTrainer:
    """
    多模态模型训练器类
    
    负责模型的训练、评估和预测，支持文本和图像的多模态训练。
    
    属性:
        model: 待训练的多模态模型
        config: 配置参数对象
        tokenizer: 文本tokenizer
        device: 训练设备（CPU/GPU）
        criterion: 损失函数（CrossEntropyLoss）
        optimizer: 优化器（Adam）
        scheduler: 学习率调整器
    """
    
    def __init__(self, model, tokenizer, config):
        """初始化训练器"""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        
        # 初始化自适应类别权重
        self.class_weights = torch.ones(config.num_classes).to(self.device)
        self.class_weights.requires_grad = False
        
        # 初始化难易样本统计
        self.class_difficulty = torch.zeros(config.num_classes).to(self.device)
        self.class_counts = torch.zeros(config.num_classes).to(self.device)
        
        # 将模型移至指定设备
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), 
                                   lr=config.learning_rate, 
                                   weight_decay=1e-4)
        
        # 学习率调整器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.2,
            patience=1,
            verbose=True
        )
        
        # 训练状态追踪
        self.early_stop_counter = 0
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.final_best_acc = 0.0
        
        # wandb配置
        self.use_wandb = config.wandb
        
        self.print_loss_temp = 0
        self.loss_type = config.loss_type 

    def adaptive_class_balanced_loss(self, outputs, labels):

        # 计算基础交叉熵损失
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        
        # 如果配置为纯交叉熵损失
        if self.loss_type == 'ce':
            return ce_loss.mean()
        
        # 原有焦点损失计算逻辑
        probs = F.softmax(outputs, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 动态调整gamma（焦点参数）
        with torch.no_grad():
            # 更新类别难度统计
            batch_class_counts = torch.bincount(labels, minlength=self.config.num_classes)
            batch_class_difficulty = torch.bincount(labels, weights=(1-pt), minlength=self.config.num_classes)
            
            self.class_counts += batch_class_counts
            self.class_difficulty += batch_class_difficulty
            
            # 计算每个类别的平均难度
            class_avg_difficulty = self.class_difficulty / (self.class_counts + 1e-8)
            
            # 修正权重计算逻辑
            log_difficulty = torch.log(class_avg_difficulty + 1.0)
            self.class_weights = 1.0 + log_difficulty * 5.0
            
            # 计算动态gamma
            gamma = 1.5 + log_difficulty[labels] * 3.0
            
            # 控制打印频率
            # if self.print_loss_temp % 100 == 0:
            #     # 计算当前batch准确率
            #     preds = torch.argmax(outputs, dim=1)
            #     batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                
            #     print(f"\n=== 第 {self.print_loss_temp + 1} 次迭代调试信息 ===")
            #     print("当前类别统计：")
            #     label_map = {0: "positive", 1: "neutral", 2: "negative"}
            #     for i in range(self.config.num_classes):
            #         print(f"{label_map[i]}: "
            #               f"count={self.class_counts[i]}, "
            #               f"difficulty={class_avg_difficulty[i]:.4f}, "
            #               f"log_difficulty={log_difficulty[i]:.4f}, "
            #               f"weight={self.class_weights[i]:.4f}")
                
            #     print("\n当前batch的pt分布：")
            #     for i in range(self.config.num_classes):
            #         mask = labels == i
            #         if mask.any():
            #             print(f"{label_map[i]}: min={pt[mask].min():.4f}, "
            #                   f"max={pt[mask].max():.4f}, "
            #                   f"mean={pt[mask].mean():.4f}")
                
            #     print("\n当前batch准确率：")
            #     print(f"整体准确率: {batch_acc:.4f}")
            #     for i in range(self.config.num_classes):
            #         mask = labels == i
            #         if mask.any():
            #             class_acc = accuracy_score(labels[mask].cpu().numpy(), preds[mask].cpu().numpy())
            #             print(f"{label_map[i]} 准确率: {class_acc:.4f}")
            
            # 更新print_loss_temp
            self.print_loss_temp += 1
        
        # 计算自适应焦点损失
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        # 增加类别间边界增强
        top2_probs, _ = torch.topk(probs, 2, dim=1)
        margin = (top2_probs[:, 0] - top2_probs[:, 1]).mean()
        boundary_loss = torch.exp(-margin * 2.0)
        
        # 应用类别权重
        weighted_loss = focal_loss * self.class_weights[labels]
        
        # 组合损失
        total_loss = weighted_loss.mean() * self.config.alpha + boundary_loss * self.config.beta
        
        # 打印损失分量（仅在控制打印频率时）
        # if (self.print_loss_temp - 1) % 100 == 0:
        #     print("\n损失分量：")
        #     print(f"基础交叉熵: {ce_loss.mean().item():.4f}")
        #     print(f"焦点损失: {focal_loss.mean().item():.4f}")
        #     print(f"边界损失: {boundary_loss.item():.4f}")
        #     print(f"总损失: {total_loss.item():.4f}")
        
        # 增加neutral类的权重
        neutral_weight = self.config.neural_init_weight  
        self.class_weights[1] = neutral_weight  # neutral类索引为1
        
        return total_loss

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        end_several_iterations_preds = []
        end_several_iterations_labels = []
        epoch_start_iteration = epoch * len(train_loader)
        end_several_iterations_loss = []
        train_loss_gap = len(train_loader) - len(train_loader)//10
        
        for batch_idx, (texts, images, labels) in enumerate(train_loader):
            # 准备数据
            texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(texts, images)
            
            # 使用新的损失函数
            loss = self.adaptive_class_balanced_loss(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
                        
            if batch_idx > train_loss_gap:
                end_several_iterations_loss.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                end_several_iterations_preds.extend(preds.cpu().numpy())
                end_several_iterations_labels.extend(labels.cpu().numpy())
            
            # 每 log_iteration 次迭代记录一次指标
            if batch_idx==0 or (batch_idx + 1) % self.config.log_iteration == 0:
                preds = torch.argmax(outputs, dim=1)
                current_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                
                if self.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_acc': current_acc,
                        'iteration': batch_idx + epoch_start_iteration + 1
                    }, step=batch_idx + epoch_start_iteration + 1)
                
                # print(f'Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] - loss: {loss.item():.4f}, acc: {current_acc:.4f}')
            # else:
            #     print(f'Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] - loss: {loss.item():.4f}')

        # 计算epoch级别的指标
        epoch_loss = np.mean(end_several_iterations_loss)
        epoch_acc = accuracy_score(end_several_iterations_labels, end_several_iterations_preds)
        
        return epoch_loss, epoch_acc

    def evaluate(self, val_loader):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # 初始化每个类别的统计
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}
        
        with torch.no_grad():
            for texts, images, labels in val_loader:
                # 准备数据
                texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(texts, images)
                loss = self.criterion(outputs, labels)
                
                # 记录损失和预测结果
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 统计每个类别的准确率
                for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        # 计算评估指标
        val_loss = total_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # 计算详细的分类报告
        report = classification_report(all_labels, all_preds, 
                                    target_names=["positive", "neutral", "negative"],
                                    digits=4,
                                    output_dict=True)
        
        # 打印详细评估指标
        print("\n详细评估指标:")
        label_map = {0: "positive", 1: "neutral", 2: "negative"}
        for label_idx in sorted(class_correct.keys()):
            metrics = report[label_map[label_idx]]
            print(f"\n{label_map[label_idx]}类别指标:")
            print(f"准确率: {class_correct[label_idx]/class_total[label_idx]:.4f} ({class_correct[label_idx]}/{class_total[label_idx]})")
            print(f"精确率: {metrics['precision']:.4f}")
            print(f"召回率: {metrics['recall']:.4f}")
            print(f"F1分数: {metrics['f1-score']:.4f}")
        
        # 打印整体指标
        print("\n整体评估指标:")
        print(f"准确率: {report['accuracy']:.4f}")
        print(f"宏平均精确率: {report['macro avg']['precision']:.4f}")
        print(f"宏平均召回率: {report['macro avg']['recall']:.4f}")
        print(f"宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
        
        return val_loss, val_acc

    def train(self, train_loader, val_loader, fold=-1):
        """执行完整的训练流程"""
        print(f"Training on {self.device}")
        
        # wandb初始化
        if self.use_wandb:
            if fold != -1:
                name = f'{self.config.name}_Multimodal_fold{fold}_iterations_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            else:
                name = f'{self.config.name}_Multimodal_iterations_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
           
            wandb.init(
                project=self.config.project_name,
                name=name,
                config=self.config.__dict__,
                reinit=True,
                allow_val_change=True
            )
            
            # wandb.watch(self.model, log="all", log_freq=self.config.log_freq)

        results = []
        start_time = time.time()

        cur_best_val_acc = 0.0
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证阶段
            val_loss, val_acc = self.evaluate(val_loader)
            
            results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # 学习率调整
            self.scheduler.step(val_acc)
            
            # 打印训练信息
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
              
            # 保存最佳模型
            if val_acc > cur_best_val_acc:
                cur_best_val_acc = val_acc
                self.best_model_state = deepcopy(self.model.state_dict())
                if val_acc > self.final_best_acc:
                    self.final_best_acc = val_acc
                print(f"Saving best model with validation accuracy: {val_acc:.4f}")
                if fold != -1:
                    torch.save(self.model.state_dict(), f"best_model_fold{fold}.pth")
                else:
                    torch.save(self.model.state_dict(), "best_model.pth")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # 早停检查
            if self.early_stop_counter >= self.config.early_stop_patience:
                print("Early stopping triggered!")
                break
                
        if self.use_wandb:
            wandb.finish()
            
        if self.use_wandb:
            if fold != -1:
                name = f'{self.config.name}_Multimodal_fold{fold}_epochs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            else:
                name = f'{self.config.name}_Multimodal_epochs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
                
            wandb.init(project=self.config.project_name, 
                        name = name,
                        config=self.config.__dict__,
                        reinit=True,
                        allow_val_change=True)
            for result in results:
                step = result['epoch']                 
                wandb.log(result, step=step)
        # 关闭wandb
        if self.use_wandb:
            wandb.finish()
            
        # 加载最佳模型
        # self.model.load_state_dict(self.best_model_state)
        return self.final_best_acc

    def predict(self, test_loader):
        """在测试集上进行预测"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for texts, images, _ in test_loader:
                texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                images = images.to(self.device)
                outputs = self.model(texts, images)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return predictions