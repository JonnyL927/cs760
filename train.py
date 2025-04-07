import os
import argparse
import subprocess

# 解析参数
parser = argparse.ArgumentParser(description="Train on Windows")
parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node")
parser.add_argument("--master_port", type=int, default=29500, help="Master port for distributed training")
args = parser.parse_args()

# 设置变量
dataset = 'timbers'
method = 'unimatch_v2'
exp = 'dinov2_small_timbers'
split = '2'

config = f'configs/{dataset}.yaml'
labeled_id_path = f'splits/{dataset}/{split}/labeled.txt'
unlabeled_id_path = f'splits/{dataset}/{split}/unlabeled.txt'
save_path = f'exp/{dataset}/{method}/{exp}/{split}'

# 创建目录
os.makedirs(save_path, exist_ok=True)

# 构建 PyTorch 分布式训练命令
cmd = [
    "python", "-m", "torch.distributed.run",
    f"--nproc_per_node={args.nproc_per_node}",
    f"--master_addr=localhost",
    f"--master_port={args.master_port}",
    f"{method}.py",
    f"--config={config}",
    f"--labeled-id-path={labeled_id_path}",
    f"--unlabeled-id-path={unlabeled_id_path}",
    f"--save-path={save_path}",
    f"--port={args.master_port}"
]

# 运行命令并记录日志
with open(os.path.join(save_path, "out.log"), "w") as log_file:
    process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
    process.communicate()
