import subprocess
import time
import sys
import os

# ================= 配置区域 =================
# 按顺序放入您要运行的脚本文件名
SCRIPTS = [
    "run_md5_full.py",
    "run_phash_full.py",
    "run_simclr_full.py"
]
# ===========================================

def run_script(script_name):
    """运行单个脚本并计时"""
    # 获取当前脚本所在目录，确保能找到同级目录下的子脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)

    if not os.path.exists(script_path):
        print(f"[错误] 找不到文件: {script_path}，已跳过。")
        return False

    print(f"\n" + "="*60)
    print(f"[启动] 正在运行: {script_name}")
    print(f"       (请勿关闭窗口，这可能需要一段时间...)")
    print("="*60 + "\n")

    start_time = time.time()
    
    try:
        # 使用当前 Python 环境运行子脚本
        # check=True 表示如果脚本报错(退出码非0)，会抛出异常
        subprocess.run([sys.executable, script_path], check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n[完成] {script_name} 运行成功！")
        print(f"耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        return True
        
    except subprocess.CalledProcessError:
        print(f"\n[失败] {script_name} 运行出错，请检查错误日志。")
        return False
    except Exception as e:
        print(f"\n[异常] 发生未知错误: {e}")
        return False

if __name__ == "__main__":
    print(f"开始批量执行 {len(SCRIPTS)} 个任务...")
    total_start = time.time()
    
    success_count = 0
    for script in SCRIPTS:
        if run_script(script):
            success_count += 1
            
    total_end = time.time()
    total_duration = total_end - total_start
    
    print("\n" + "#"*60)
    print(f"所有任务结束！")
    print(f"成功: {success_count} / {len(SCRIPTS)}")
    print(f"总耗时: {total_duration/60:.2f} 分钟")
    print(f"结果文件请查看 result 文件夹")
    print("#"*60)