import os
import itertools
import subprocess

# 定义要搜索的超参数空间
param_grid = {
    'loss_type': ['acb', 'ce'],  # 损失函数类型
    'alpha': [0.25, 0.5, 0.75, 0.9],    # focal损失的alpha参数
    'neural_init_weight': [0.5, 1.0, 1.5],  # neutral类的初始权重
    'dropout': [0.15, 0.2, 0.25, 0.3, 0.35]     # dropout率
}

# 生成所有可能的参数组合
param_combinations = list(itertools.product(
    param_grid['loss_type'],
    param_grid['alpha'],
    param_grid['neural_init_weight'],
    param_grid['dropout']
))

# 记录最佳结果
best_accuracy = 0
best_params = None

# 遍历所有参数组合
for i, (loss_type, alpha, neural_init_weight, dropout) in enumerate(param_combinations):
    # 计算beta值
    beta = 1 - alpha
    print(f"\n正在测试组合 {i+1}/{len(param_combinations)}")
    print(f"参数: loss_type={loss_type}, alpha={alpha}, beta={beta}, neural_init_weight={neural_init_weight}, dropout={dropout}")
    
    # 根据参数生成实验名称
    exp_name = f"loss_{loss_type}_alpha{alpha}_beta{beta}_weight{neural_init_weight}_dropout{dropout}"
    
    # 构建命令行参数
    cmd = [
        'python', 'main.py',
        '--loss_type', loss_type,
        '--alpha', str(alpha),
        '--beta', str(beta),
        '--neural_init_weight', str(neural_init_weight),
        '--dropout', str(dropout),
        '--name', exp_name,
        '--wandb', 'True'
    ]
    
    # 运行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 创建日志目录
    log_dir = "loss_search_logs"
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
                    'loss_type': loss_type,
                    'alpha': alpha,
                    'beta': beta,
                    'neural_init_weight': neural_init_weight,
                    'dropout': dropout
                }
            break

# 输出最佳结果
print("\n最佳参数组合:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"最佳验证准确率: {best_accuracy:.4f}")