import os
import itertools
import subprocess

# 定义要搜索的超参数空间
param_grid = {
    'feature_fusion': ['concat', 'attention'],
    'text_dim': [128, 256, 512],
    'image_dim': [64, 128, 256],
    'learning_rate': [1e-4, 1e-5],
    'dropout': [0.1, 0.3, 0.5],
    'batch_size': [32]
}

# 生成所有可能的参数组合
param_combinations = list(itertools.product(
    param_grid['feature_fusion'],
    param_grid['text_dim'],
    param_grid['image_dim'],
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['dropout']
))

# 记录最佳结果
best_accuracy = 0
best_params = None

# 遍历所有参数组合
for i, (fusion, text_dim, img_dim, lr, bs, dropout) in enumerate(param_combinations):
    print(f"\n正在测试组合 {i+1}/{len(param_combinations)}")
    print(f"参数: fusion={fusion}, text_dim={text_dim}, img_dim={img_dim}, lr={lr}, bs={bs}, dropout={dropout}")
    
    # 根据参数生成实验名称
    exp_name = f"fusion_{fusion}_text{text_dim}_img{img_dim}_lr{lr}_bs{bs}_dropout{dropout}"
    
    # 构建命令行参数
    cmd = [
        'python', 'main.py',
        '--feature_fusion', fusion,
        '--text_dim', str(text_dim),
        '--image_dim', str(img_dim),
        '--learning_rate', str(lr),
        '--batch_size', str(bs),
        '--dropout', str(dropout),
        '--name', exp_name,  # 添加实验名称
        '--wandb', 'True'
    ]
    
    # 运行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 创建日志目录
    log_dir = "search_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 将输出写入日志文件
    log_file = os.path.join(log_dir, f"{exp_name}.txt")
    with open(log_file, 'w') as f:
        f.write("=== 命令 ===\n")
        f.write(' '.join(cmd) + '\n\n')
        f.write("=== 标准输出 ===\n")
        f.write(result.stdout + '\n')
        f.write("=== 标准错误 ===\n")
        f.write(result.stderr + '\n')
    
    # 从输出中提取验证准确率
    output = result.stdout
    for line in output.split('\n'):
        if 'Average Validation Accuracy' in line:
            accuracy = float(line.split(': ')[1])
            print(f"验证准确率: {accuracy:.4f}")
            
            # 更新最佳结果
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'feature_fusion': fusion,
                    'text_dim': text_dim,
                    'image_dim': img_dim,
                    'learning_rate': lr,
                    'batch_size': bs,
                    'dropout': dropout
                }
            break

# 输出最佳结果
print("\n最佳参数组合:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"最佳验证准确率: {best_accuracy:.4f}")