import psutil
import GPUtil
import time
import os
import socket

def get_gpu_utilization(gpu_id):
    gpus = GPUtil.getGPUs()
    if not gpus or gpu_id >= len(gpus):
        return 0
    return gpus[gpu_id].load * 100

def main():
    # 检查并创建 'result' 文件夹
    if not os.path.exists('result'):
        os.mkdir('result')
    
    # 获取节点名
    hostname = socket.gethostname()
    
    # 分别为0号和1号GPU创建文件
    with open(f"result/utilization_data_{hostname}_GPU0.txt", "a") as f0, \
         open(f"result/utilization_data_{hostname}_GPU1.txt", "a") as f1:
        f0.write("timestamp,cpu_utilization,gpu0_utilization\n")
        f1.write("timestamp,cpu_utilization,gpu1_utilization\n")
        
        count = 0
        while True:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            cpu_util = psutil.cpu_percent(interval=1)
            gpu0_util = get_gpu_utilization(0)
            gpu1_util = get_gpu_utilization(1)
            
            f0.write(f"{timestamp},{cpu_util},{gpu0_util}\n")
            f1.write(f"{timestamp},{cpu_util},{gpu1_util}\n")
            
            count += 1
            if count % 10 == 0:  # 每10次刷新一次
                f0.flush()
                f1.flush()
            time.sleep(1)

if __name__ == "__main__":
    main()
