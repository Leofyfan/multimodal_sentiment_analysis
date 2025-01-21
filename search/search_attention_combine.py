import os
import subprocess

def search_attention_combine_params():
    """测试attention_combine融合方式的超参数搜索"""
    # 定义要搜索的超参数空间
    param_grid = {
        'text_dim': [128, 256, 512],  # 文本维度
        'dropout': [0.1, 0.15, 0.2],  # dropout率
        'learning_rate': [8e-5, 1e-5, 5e-6],  # 学习率
        'feature_fusion': ['attention_combine']  # 固定使用attention_combine融合方式
    }

    # 记录最佳结果
    best_accuracy = 0
    best_params = None

    # 遍历所有参数组合
    total_combinations = len(param_grid['text_dim']) * \
                        len(param_grid['dropout']) * len(param_grid['learning_rate'])
    current_combination = 0

    for text_dim in param_grid['text_dim']:
        for dropout in param_grid['dropout']:
            for lr in param_grid['learning_rate']:
                current_combination += 1
                print(f"\n正在测试组合 {current_combination}/{total_combinations}")
                print(f"参数: text_dim={text_dim}, "
                     f"dropout={dropout}, lr={lr}, fusion=attention_combine")
                
                # 根据参数生成实验名称
                exp_name = f"attention_combine_textdim{text_dim}_dropout{dropout}_lr{lr}"
                
                # 获取当前脚本所在目录
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # 获取上一级目录路径
                project_root = os.path.dirname(script_dir)
                # 获取main.py的绝对路径
                main_py_path = os.path.join(project_root, 'main.py')
                
                # 构建命令行参数
                cmd = [
                    'python', main_py_path,
                    '--text_dim', str(text_dim),
                    '--image_dim', str(text_dim),
                    '--dropout', str(dropout),
                    '--learning_rate', str(lr),
                    '--feature_fusion', 'attention_combine',
                    '--name', exp_name,
                    '--wandb', 'True'
                ]
                
                # 检查main.py是否存在
                if not os.path.exists(main_py_path):
                    raise FileNotFoundError(f"main.py 文件未找到，路径: {main_py_path}")
                
                # 运行命令并捕获输出
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # 创建日志目录
                log_dir = os.path.join(script_dir, "../logs/attention_combine_search_logs")
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
                                    'text_dim': text_dim,
                                    'dropout': dropout,
                                    'learning_rate': lr,
                                    'feature_fusion': 'attention_combine'
                                }
                            break
                        except (IndexError, ValueError) as e:
                            print(f"解析准确率时出错: {e}")
                            continue
                
                if not accuracy_found:
                    print("警告：未找到验证准确率，请检查模型输出")

    # 输出最佳结果
    if best_params is not None:
        print("\n最佳参数组合:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"最佳验证准确率: {best_accuracy:.4f}")
    else:
        print("\n警告：未找到有效的验证准确率，请检查模型输出和日志文件")

if __name__ == "__main__":
    search_attention_combine_params()