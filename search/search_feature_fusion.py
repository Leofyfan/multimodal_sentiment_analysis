"""
特征融合方式的参数搜索
1. 使用search_concat_params()函数进行简单拼接的参数搜索
2. 使用search_combine_params()函数进行特征组合的参数搜索
3. 使用search_attention_params()函数进行注意力机制的参数搜索
4. 使用search_attention_concat_params()函数进行注意力机制和简单拼接的参数搜索
5. 使用search_attention_combine_params()函数进行注意力机制和特征组合的参数搜索
6. 使用search_encoder_params()函数进行编码器融合的参数搜索
"""

import os
import subprocess
from search_concat import search_concat_params
from search_combine import search_combine_params
from search_attention import search_attention_params
from search_attention_concat import search_attention_concat_params
from search_attention_combine import search_attention_combine_params
from search_encoder import search_encoder_params

if __name__ == "__main__":
    print("="*100)
    print("开始进行融合方式参数搜索...")
    print("="*100)
    
    search_functions = [
        search_attention_concat_params,
        search_encoder_params
    ]
    
    for func in search_functions:
        try:
            func()
        except Exception as e:
            print(f"函数 {func.__name__} 执行出错: {str(e)}")
            continue
    
    print("="*100)
    print("所有融合方式参数搜索完成！")
    print("="*100)
