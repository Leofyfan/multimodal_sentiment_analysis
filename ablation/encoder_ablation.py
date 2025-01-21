import os
import subprocess

def encoder_ablation():
    """基于transformer-encoder的融合方式的模态消融实验"""
    # 定义要测试的模态组合
    modality_combinations = [
        # {'use_text': True, 'use_image': False, 'text_dim': 256, 'dropout': 0.15, 'learning_rate': 5e-06},  # 仅文本
        # {'use_text': False, 'use_image': True, 'text_dim': 256, 'dropout': 0.15, 'learning_rate': 5e-06},  # 仅图像
        {'use_text': True, 'use_image': True, 'text_dim': 256, 'dropout': 0.15, 'learning_rate': 5e-06}    # 文本+图像
    ]

    # 记录最佳结果
    best_accuracy = 0
    best_params = None

    # 遍历所有模态组合
    for idx, modality in enumerate(modality_combinations):
        print(f"\n正在测试组合 {idx+1}/{len(modality_combinations)}")
        print(f"参数: use_text={modality['use_text']}, use_image={modality['use_image']}, text_dim={modality['text_dim']}, dropout={modality['dropout']}, learning_rate={modality['learning_rate']}")
        
        # 根据参数生成实验名称
        exp_name = f"encoder_text{modality['use_text']}_image{modality['use_image']}_textdim{modality['text_dim']}_dropout{modality['dropout']}_lr{modality['learning_rate']}"
        
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上一级目录路径
        project_root = os.path.dirname(script_dir)
        # 获取main.py的绝对路径
        main_py_path = os.path.join(project_root, 'main.py')
        
        # 构建命令行参数
        cmd = [
            'python', main_py_path,
            '--feature_fusion', 'encoder',
            '--use_text', '1' if modality['use_text'] else '0',
            '--use_image', '1' if modality['use_image'] else '0',
            '--text_dim', str(modality['text_dim']),
            '--image_dim', str(modality['text_dim']),
            '--dropout', str(modality['dropout']),
            '--learning_rate', str(modality['learning_rate']),
            '--name', exp_name,
            '--wandb', 'True'
        ]
        
        # 检查main.py是否存在
        if not os.path.exists(main_py_path):
            raise FileNotFoundError(f"main.py 文件未找到，路径: {main_py_path}")
        
        # 运行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 创建日志目录
        log_dir = os.path.join(script_dir, "../logs/ablation_encoder_logs")
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
        accuracy_found = False
        for line in output.split('\n'):
            if 'Best validation accuracy:' in line:
                try:
                    accuracy = float(line.split(': ')[1].strip())
                    print(f"验证准确率: {accuracy:.4f}")
                    accuracy_found = True
                    
                    # 更新最佳结果
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'use_text': modality['use_text'],
                            'use_image': modality['use_image'],
                            'text_dim': modality['text_dim'],
                            'dropout': modality['dropout'],
                            'learning_rate': modality['learning_rate']
                        }
                    break
                except (IndexError, ValueError) as e:
                    print(f"解析准确率时出错: {e}")
                    continue
        
        if not accuracy_found:
            print("警告：未找到验证准确率，请检查模型输出")

    # 输出最佳结果
    if best_params is not None:
        print("\n最佳模态组合:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"最佳验证准确率: {best_accuracy:.4f}")
    else:
        print("\n警告：未找到有效的验证准确率，请检查模型输出和日志文件")

        
if __name__ == "__main__":
    encoder_ablation()
