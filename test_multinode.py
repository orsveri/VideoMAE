# test_multinode.py
import os
import socket
import torch.distributed as dist

def init_process(rank, size):
    """ Initialize distributed environment """
    hostname = socket.gethostname()
    print(f"Process {rank} running on {hostname}")
    dist.init_process_group("nccl", rank=rank, world_size=size)
    print(f"Process {rank} initialized.")

if __name__ == "__main__":
    rank0 = int(os.environ['OMPI_COMM_WORLD_RANK'])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    init_process(rank, world_size)
