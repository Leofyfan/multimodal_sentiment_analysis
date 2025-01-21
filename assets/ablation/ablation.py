"""
模态消融实验代码
1. 使用concat_ablation()函数进行简单拼接的消融实验
2. 使用combine_ablation()函数进行特征组合的消融实验
3. 使用attention_ablation()函数进行注意力机制的消融实验
4. 使用attention_concat_ablation()函数进行注意力机制和简单拼接的消融实验
5. 使用attention_combine_ablation()函数进行注意力机制和特征组合的消融实验
6. 使用encoder_ablation()函数进行编码器融合的消融实验
"""

import os
import subprocess
from concat_ablation import concat_ablation
from combine_ablation import combine_ablation
from attention_ablation import attention_ablation
from attention_concat_ablation import attention_concat_ablation
from attention_combine_ablation import attention_combine_ablation
from encoder_ablation import encoder_ablation


if __name__ == "__main__":
    # concat_ablation()
    # combine_ablation()
    # attention_ablation()
    # attention_concat_ablation()
    attention_combine_ablation()
    encoder_ablation()
