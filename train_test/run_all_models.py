import os
import subprocess
import time
from datetime import datetime

def run_model(model_name, script_path):
    """运行指定模型的训练脚本并记录时间（在 conda 环境 yclearning 中执行）"""
    print(f"\n{'='*50}")
    print(f"开始运行 {model_name} 模型训练...")
    print("将使用 Conda 环境: yclearning")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    script_dir = os.path.dirname(script_path)
    script_file = os.path.basename(script_path)
    
    # 优先使用 conda run 调用指定环境（无缓冲输出，且不由 conda 捕获输出）
    cmd = ['conda', 'run', '--no-capture-output', '-n', 'yclearning', 'python', '-u', script_file]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        # 强制子进程以 UTF-8 输出，防止 Windows 控制台 GBK 解码报错
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout，保证实时可见
            cwd=script_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env
        )
    except FileNotFoundError:
        # 回退方案：如果找不到 conda，则直接使用系统 python（无缓冲）
        print("警告: 未找到 conda，可执行文件不在 PATH？改用系统 Python 直接运行。")
        fallback_cmd = ['python', '-u', script_file]
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            fallback_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=script_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env
        )
    
    # 实时输出脚本执行结果（逐行）
    for line in process.stdout:
        if line:
            print(line.rstrip(), flush=True)
    
    # 等待结束
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{model_name} 模型训练完成!")
    print(f"耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"{'='*50}\n")
    
    return duration

def main():
    # 基于脚本自身位置定位（无需依赖当前工作目录）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要运行的模型列表
    models = [
        {
            'name': '动脉模型',
            'path': os.path.join(base_dir, 'artery', 'code', 'simple1_gpu.py')
        },
        {
            'name': '眼科模型',
            'path': os.path.join(base_dir, 'eye', 'code', 'simple1_gpu.py')
        },
        {
            'name': '生理模型',
            'path': os.path.join(base_dir, 'physiology', 'code', 'simple1_gpu.py')
        },
        {
            'name': '全量模型',
            'path': os.path.join(base_dir, 'all', 'code', 'simple1_gpu.py')
        }
    ]
    
    # 记录总体开始时间
    total_start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n开始顺序运行所有模型训练 - {start_datetime}")
    print("="*60)
    
    # 顺序运行每个模型
    model_times = {}
    for model in models:
        duration = run_model(model['name'], model['path'])
        model_times[model['name']] = duration
    
    # 计算总耗时
    total_duration = time.time() - total_start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 打印总结报告
    print("\n训练完成总结报告")
    print("="*60)
    print(f"开始时间: {start_datetime}")
    print(f"结束时间: {end_datetime}")
    print(f"\n各模型耗时:")
    for name, duration in model_times.items():
        print(f"{name}: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"\n总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print("="*60)

if __name__ == "__main__":
    main()
