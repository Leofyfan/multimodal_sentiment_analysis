import os
import subprocess

def loss_ablation():
    """测试不同融合方式下损失函数的消融实验"""
    # 定义要测试的融合方式和损失函数组合（使用各自最优参数配置）
    fusion_loss_combinations = [
        # 简单拼接方式
        {
            'feature_fusion': 'concat',
            'loss_type': 'acb',
            'text_dim': 256,
            'image_dim': 128,
            'dropout': 0.25,
            'learning_rate': 0.0001
        },
        {
            'feature_fusion': 'concat',
            'loss_type': 'ce',
            'text_dim': 256,
            'image_dim': 128,
            'dropout': 0.25,
            'learning_rate': 0.0001
        },
        
        # 特征组合
        {
            'feature_fusion': 'combine',
            'loss_type': 'acb',
            'text_dim': 256,
            'dropout': 0.25,
            'learning_rate': 0.0001
        },
        {
            'feature_fusion': 'combine',
            'loss_type': 'ce',
            'text_dim': 256,
            'dropout': 0.25,
            'learning_rate': 0.0001
        },
        
        # 编码器融合
        {
            'feature_fusion': 'encoder',
            'loss_type': 'acb',
            'text_dim': 256,
            'dropout': 0.15,
            'learning_rate': 5e-06
        },
        {
            'feature_fusion': 'encoder',
            'loss_type': 'ce',
            'text_dim': 256,
            'dropout': 0.15,
            'learning_rate': 5e-06
        },
        
        # 注意力机制
        {
            'feature_fusion': 'attention',
            'loss_type': 'acb',
            'text_dim': 512,
            'dropout': 0.2,
            'learning_rate': 2e-05
        },
        {
            'feature_fusion': 'attention',
            'loss_type': 'ce',
            'text_dim': 512,
            'dropout': 0.2,
            'learning_rate': 2e-05
        },
        
        # 注意力拼接
        {
            'feature_fusion': 'attention_concat',
            'loss_type': 'acb',
            'text_dim': 128,
            'image_dim': 128,
            'dropout': 0.15,
            'learning_rate': 8e-05
        },
        {
            'feature_fusion': 'attention_concat',
            'loss_type': 'ce',
            'text_dim': 128,
            'image_dim': 128,
            'dropout': 0.15,
            'learning_rate': 8e-05
        },
        
        # 注意力组合
        {
            'feature_fusion': 'attention_combine',
            'loss_type': 'acb',
            'text_dim': 128,
            'dropout': 0.15,
            'learning_rate': 8e-05
        },
        {
            'feature_fusion': 'attention_combine',
            'loss_type': 'ce',
            'text_dim': 128,
            'dropout': 0.15,
            'learning_rate': 8e-05
        }
    ]

    # 记录最佳结果
    best_accuracy = 0
    best_params = None

    # 遍历所有组合
    for idx, params in enumerate(fusion_loss_combinations):
        print(f"\n正在测试组合 {idx+1}/{len(fusion_loss_combinations)}")
        print(f"参数: feature_fusion={params['feature_fusion']}, loss_type={params['loss_type']}")
        
        # 根据参数生成实验名称
        exp_name = f"loss_ablation_{params['feature_fusion']}_{params['loss_type']}"
        
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上一级目录路径
        project_root = os.path.dirname(script_dir)
        # 获取main.py的绝对路径
        main_py_path = os.path.join(project_root, 'main.py')
        
        # 构建命令行参数
        cmd = [
            'python', main_py_path,
            '--feature_fusion', params['feature_fusion'],
            '--loss_type', params['loss_type'],
            '--text_dim', str(params['text_dim']),
            '--dropout', str(params['dropout']),
            '--learning_rate', str(params['learning_rate']),
            '--name', exp_name,
            '--wandb', 'True'
        ]
        
        # 对于需要image_dim的融合方式，添加相应参数
        if 'image_dim' in params:
            cmd.extend(['--image_dim', str(params['image_dim'])])
        
        # 检查main.py是否存在
        if not os.path.exists(main_py_path):
            raise FileNotFoundError(f"main.py 文件未找到，路径: {main_py_path}")
        
        # 运行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 创建日志目录
        log_dir = os.path.join(script_dir, "../logs/ablation_loss_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 将输出写入日志文件
        log_file = os.path.join(log_dir, f"{exp_name}.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=== 命令 ===\n")
            f.write(' '.join(cmd) + '\n\n')
            f.write("=== 标准输出 ===\n")
            f.write(result.stdout + '\n')
            f.write("=== wandb 日志 ===\n")
            f.write(result.stderr + '\n')
        
        # 从输出中提取验证准确率
        output = result.stdout
        metrics_found = False
        
        for line in output.split('\n'):
            if 'Best validation accuracy:' in line:
                try:
                    accuracy = float(line.split(': ')[1].strip())
                    print(f"\n验证准确率: {accuracy:.4f}")
                    metrics_found = True
                    
                    # 更新最佳结果
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = params.copy()
                    break
                except (IndexError, ValueError) as e:
                    print(f"解析指标时出错: {e}")
                    continue
        
        if not metrics_found:
            print("警告：未找到评估指标，请检查模型输出")

    # 输出最佳结果
    if best_params is not None:
        print("\n最佳组合:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"最佳验证准确率: {best_accuracy:.4f}")
    else:
        print("\n警告：未找到有效的评估指标，请检查模型输出和日志文件")

if __name__ == "__main__":
    loss_ablation()