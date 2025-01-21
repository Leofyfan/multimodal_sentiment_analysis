"""
该模块实现了多模态数据集的加载和处理功能。

主要功能：
1. 支持图像和文本的多模态数据加载
2. 实现数据增强策略（文本增强和图像增强）
3. 提供数据集统计信息的计算和展示
4. 支持样本权重计算用于不平衡数据处理
"""

import os
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torch
import random
from nltk.corpus import wordnet
from torchvision import transforms
from PIL import ImageOps, ImageEnhance

# 数据集类
class MultimodalDataset(Dataset):
    """
    多模态数据集类，用于加载和处理图像-文本对数据。
    
    继承自 torch.utils.data.Dataset，实现了数据集的基本功能。
    
    属性:
        data_dir: 数据目录路径
        transform: 图像转换函数
        is_train: 是否为训练模式
        augment_ratio: 数据增强比例
        guid_list: 数据标识符列表
        label_list: 标签列表
        data_stats: 数据集统计信息
    """
    
    def __init__(self, data_dir, data_file, transform=None, is_train=True, augment_ratio=2.0):
        """
        初始化多模态数据集
        
        参数:
            data_dir: 数据目录路径
            data_file: 数据文件路径
            transform: 图像转换函数
            is_train: 是否为训练模式
            augment_ratio: 数据增强比例
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.augment_ratio = augment_ratio
        self.guid_list, self.label_list = self._load_data(data_file)
        
        # 数据增强相关初始化
        if self.is_train:
            self._prepare_augmented_data()
        
        self.data_stats = self._calculate_stats()
        
    def _load_data(self, data_file):
        """
        从文件加载数据
        
        参数:
            data_file: 数据文件路径
            
        返回:
            tuple: (guid列表, 标签列表)
        """
        guid_list, label_list = [], []
        with open(data_file, "r") as f:
            lines = f.readlines()[1:]  # 跳过标题行
            for line in lines:
                guid, tag = line.strip().split(",")
                guid_list.append(guid)
                label_list.append(tag)
        return guid_list, label_list

    def _calculate_stats(self):
        """
        计算数据集统计信息
        
        返回:
            dict: 包含样本总数、类别分布等统计信息
        """
        stats = {
            "total_samples": len(self.guid_list),
            "original_samples": 0,
            "augmented_samples": 0,
            "label_distribution": {},
            "missing_text": 0,
            "missing_image": 0
        }
        
        for guid, label in zip(self.guid_list, self.label_list):
            # 区分原始数据和增强数据
            if "_aug" in guid:
                stats["augmented_samples"] += 1
            else:
                stats["original_samples"] += 1
            
            # 统计标签分布
            if label not in stats["label_distribution"]:
                stats["label_distribution"][label] = 0
            stats["label_distribution"][label] += 1
            
            # 只检查原始数据的文件存在性
            if "_aug" not in guid:
                # 检查文本文件
                text_path = os.path.join(self.data_dir, f"{guid}.txt")
                if not os.path.exists(text_path):
                    stats["missing_text"] += 1
                    
                # 检查图像文件
                image_path = os.path.join(self.data_dir, f"{guid}.jpg")
                if not os.path.exists(image_path):
                    stats["missing_image"] += 1
                
        return stats

    def print_stats(self):
        """打印数据集统计信息"""
        print("\n数据集统计信息:")
        print(f"总样本数: {self.data_stats['total_samples']}")
        print(f"原始样本数: {self.data_stats['original_samples']}")
        print(f"增强样本数: {self.data_stats['augmented_samples']}")
        print("\n标签分布:")
        for label, count in self.data_stats["label_distribution"].items():
            percentage = (count / self.data_stats['total_samples']) * 100
            print(f"{label}: {count} ({percentage:.2f}%)")
        print(f"\n缺失文本数: {self.data_stats['missing_text']}")
        print(f"缺失图像数: {self.data_stats['missing_image']}")

    def __len__(self):
        return len(self.guid_list)

    def _prepare_augmented_data(self):
        """
        准备增强数据
        
        功能：
        1. 计算每个类别需要增强的数量
        2. 特别增加neutral和negative样本
        3. 生成增强样本的标识符
        """
        original_size = len(self.guid_list)
        target_size = int(original_size * self.augment_ratio)
        
        # 计算每个类别需要增强的数量
        label_counts = {}
        for label in self.label_list:
            label_counts[label] = label_counts.get(label, 0) + 1
            
        # 特别增加neutral和negative样本
        target_counts = {
            "positive": target_size // 3,
            "neutral": target_size // 3,
            "negative": target_size // 3
        }
        
        # 收集需要增强的样本
        augmented_samples = []
        for idx, (guid, label) in enumerate(zip(self.guid_list, self.label_list)):
            if label_counts[label] < target_counts[label]:
                # 对需要增强的类别进行多次复制和增强
                repeat_times = max(1, int(target_counts[label] / label_counts[label]))
                for _ in range(repeat_times):
                    augmented_samples.append((f"{guid}_aug{_}", label))
        
        # 添加增强样本
        for aug_guid, label in augmented_samples:
            self.guid_list.append(aug_guid)
            self.label_list.append(label)
            
    def _text_augment(self, text):
        """
        文本数据增强
        
        实现的增强方法：
        1. 同义词替换
        2. 随机插入
        3. 随机删除
        
        参数:
            text: 原始文本
            
        返回:
            str: 增强后的文本
        """
        # 同义词替换
        def synonym_replacement(words, n=1):
            new_words = words.copy()
            random_word_list = list(set([word for word in words]))
            random.shuffle(random_word_list)
            num_replaced = 0
            for random_word in random_word_list:
                synonyms = self._get_synonyms(random_word)
                if len(synonyms) >= 1:
                    synonym = random.choice(synonyms)
                    new_words = [synonym if word == random_word else word for word in new_words]
                    num_replaced += 1
                if num_replaced >= n: 
                    break
            return new_words

        # 随机插入
        def random_insertion(words, n=1):
            new_words = words.copy()
            for _ in range(n):
                self._add_word(new_words)
            return new_words

        # 随机删除
        def random_deletion(words, p=0.1):
            if len(words) == 1:
                return words
            new_words = []
            for word in words:
                if random.uniform(0, 1) > p:
                    new_words.append(word)
            if len(new_words) == 0:
                return [random.choice(words)]
            return new_words

        words = text.split()
        if len(words) < 2:
            return text
            
        # 随机选择一种增强方式
        augment_method = random.choice([
            lambda x: synonym_replacement(x, n=1),
            lambda x: random_insertion(x, n=1),
            lambda x: random_deletion(x, p=0.1)
        ])
        
        try:
            new_words = augment_method(words)
            return " ".join(new_words)
        except:
            return text
            
    def _image_augment(self, image):
        """
        图像数据增强
        
        实现的增强方法：
        1. 自动对比度调整
        2. 亮度调整
        3. 对比度调整
        4. 随机旋转
        5. 直方图均衡化
        
        参数:
            image: PIL图像对象
            
        返回:
            Image: 增强后的图像
        """
        # 随机选择一种增强方式
        augment_method = random.choice([
            lambda img: ImageOps.autocontrast(img),
            lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
            lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
            lambda img: img.rotate(random.randint(-15, 15)),
            lambda img: ImageOps.equalize(img)
        ])
        
        try:
            return augment_method(image)
        except:
            return image
            
    def _get_synonyms(self, word):
        """
        获取单词的同义词
        
        参数:
            word: 输入单词
            
        返回:
            list: 同义词列表
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)
        
    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        处理步骤：
        1. 加载文本和图像数据
        2. 对增强样本进行数据增强
        3. 转换标签格式
        
        参数:
            idx: 数据索引
            
        返回:
            tuple: (文本, 图像, 标签)
        """
        guid = self.guid_list[idx]
        label = self.label_list[idx]
        MAX_LEN = 128

        # 判断是否是增强样本
        is_augmented = "_aug" in guid
        original_guid = guid.split("_aug")[0]

        # 加载文本
        text_path = os.path.join(self.data_dir, f"{original_guid}.txt")
        try:
            with open(text_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()[:MAX_LEN]
                if is_augmented and self.is_train:
                    text = self._text_augment(text)
        except Exception as e:
            print(f"警告：读取文件 {text_path} 时出错: {str(e)}")
            text = ""

        # 加载图像
        image_path = os.path.join(self.data_dir, f"{original_guid}.jpg")
        image = Image.open(image_path).convert("RGB")
        if is_augmented and self.is_train:
            image = self._image_augment(image)
        if self.transform:
            image = self.transform(image)

        # 修改标签处理逻辑
        if label == "null":
            label = -1
        else:
            label_map = {"positive": 0, "neutral": 1, "negative": 2}
            label = label_map.get(label, -1)
        
        return text, image, label

    def get_sample_weights(self):
        """
        计算每个样本的权重
        
        用于处理类别不平衡问题
        
        返回:
            tensor: 样本权重张量
        """
        label_counts = self.data_stats["label_distribution"]
        weights = []
        for label in self.label_list:
            weights.append(1.0 / label_counts[label])
        return torch.tensor(weights)