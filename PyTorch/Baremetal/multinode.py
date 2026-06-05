import os
import time
import socket

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class SyntheticImageDataset(Dataset):
    def __init__(self, dataset_size, num_classes, image_size):
        self.dataset_size = dataset_size
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


# ------------------------------------------------------------
# DDP INITIALIZATION — MPI launch, PyTorch env:// style
# ------------------------------------------------------------

rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

# Match the other group's style:
# each MPI process sees only its assigned local GPU.
#
# After this, the assigned GPU becomes cuda:0 inside this process.
# This must happen before any CUDA call.
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

# Provide standard PyTorch distributed environment variables.
os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world_size)

torch.cuda.set_device(0)
device = torch.device("cuda:0")

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size,
    device_id=device,
)

hostname = socket.gethostname()

if rank == 0:
    print("===== Multi-Node Multi-GPU ResNet-50 DDP Benchmark =====")
    print("Init style: MPI env -> PyTorch env://")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"NCCL version: {torch.cuda.nccl.version()}")
    print(f"World size / total GPUs: {world_size}")
    print(f"Master address: {os.environ.get('MASTER_ADDR')}")
    print(f"Master port: {os.environ.get('MASTER_PORT')}")
    print()

print(
    f"[Rank {rank}] Host: {hostname}, "
    f"OMPI local rank: {local_rank}, "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
    f"Process CUDA device: {torch.cuda.current_device()}, "
    f"GPU: {torch.cuda.get_device_name(device)}",
    flush=True,
)

dist.barrier()


# ------------------------------------------------------------
# WORKLOAD PARAMETERS
# ------------------------------------------------------------

dataset_size = 2000000
train_ratio = 0.8
num_classes = 10
image_size = 224
batch_size = 1024
epochs = 20
num_workers = 4

if rank == 0:
    print("===== Workload Parameters =====")
    print(f"Dataset size: {dataset_size}")
    print(f"Train/test split: {int(train_ratio * 100)}% / {int((1 - train_ratio) * 100)}%")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Per-GPU batch size: {batch_size}")
    print(f"Effective global batch size: {batch_size * world_size}")
    print(f"Epochs: {epochs}")
    print(f"DataLoader workers per process: {num_workers}")
    print()


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ------------------------------------------------------------
# DATASET + DISTRIBUTED SAMPLER
# ------------------------------------------------------------

dataset = SyntheticImageDataset(dataset_size, num_classes, image_size)

train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,
)

test_sampler = DistributedSampler(
    test_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    drop_last=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Because each rank sees only one GPU, that GPU is cuda:0.
model = DDP(
    model,
    device_ids=[0],
    output_device=0,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ------------------------------------------------------------
# SYNCHRONIZE BEFORE TIMING
# ------------------------------------------------------------

torch.cuda.reset_peak_memory_stats()
dist.barrier()
torch.cuda.synchronize()

total_train_samples_local = 0
train_start_time = time.time()


# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------

if rank == 0:
    print("===== Training =====")

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    model.train()

    epoch_start_time = time.time()

    running_loss = 0.0
    epoch_samples_local = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_samples = inputs.size(0)
        running_loss += loss.item() * batch_samples

        epoch_samples_local += batch_samples
        total_train_samples_local += batch_samples

    torch.cuda.synchronize()
    dist.barrier()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    epoch_samples_tensor = torch.tensor(
        epoch_samples_local,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(epoch_samples_tensor, op=dist.ReduceOp.SUM)
    epoch_samples_global = epoch_samples_tensor.item()

    loss_tensor = torch.tensor(
        running_loss,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    epoch_loss = loss_tensor.item() / epoch_samples_global
    epoch_throughput = epoch_samples_global / epoch_time

    if rank == 0:
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train loss: {epoch_loss:.4f}")
        print(f"  Epoch time: {epoch_time:.4f} seconds")
        print(f"  Global train throughput: {epoch_throughput:.2f} samples/sec")


torch.cuda.synchronize()
dist.barrier()

train_end_time = time.time()
total_train_time = train_end_time - train_start_time

total_train_samples_tensor = torch.tensor(
    total_train_samples_local,
    dtype=torch.float64,
    device=device,
)

dist.all_reduce(total_train_samples_tensor, op=dist.ReduceOp.SUM)

total_train_samples_global = total_train_samples_tensor.item()
overall_train_throughput = total_train_samples_global / total_train_time


# ------------------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------------------

if rank == 0:
    print("===== Evaluation =====")

model.eval()

test_loss_local = 0.0
correct_local = 0
total_test_samples_local = 0

dist.barrier()
torch.cuda.synchronize()

eval_start_time = time.time()

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_samples = inputs.size(0)

        test_loss_local += loss.item() * batch_samples
        total_test_samples_local += batch_samples

        predicted = outputs.argmax(dim=1)
        correct_local += (predicted == targets).sum().item()

torch.cuda.synchronize()
dist.barrier()

eval_end_time = time.time()
eval_time = eval_end_time - eval_start_time

test_loss_tensor = torch.tensor(test_loss_local, dtype=torch.float64, device=device)
correct_tensor = torch.tensor(correct_local, dtype=torch.float64, device=device)
test_samples_tensor = torch.tensor(total_test_samples_local, dtype=torch.float64, device=device)

dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
dist.all_reduce(test_samples_tensor, op=dist.ReduceOp.SUM)

total_test_samples_global = test_samples_tensor.item()
avg_test_loss = test_loss_tensor.item() / total_test_samples_global
test_accuracy = correct_tensor.item() / total_test_samples_global
eval_throughput = total_test_samples_global / eval_time

peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3


# ------------------------------------------------------------
# FINAL RESULTS
# ------------------------------------------------------------

if rank == 0:
    print("===== Final Results =====")
    print(f"Total GPUs / world size: {world_size}")
    print(f"Total training time: {total_train_time:.4f} seconds")
    print(f"Global training throughput: {overall_train_throughput:.2f} samples/sec")
    print(f"Evaluation time: {eval_time:.4f} seconds")
    print(f"Evaluation throughput: {eval_throughput:.2f} samples/sec")
    print(f"Final test loss: {avg_test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Peak GPU memory allocated per rank: {peak_memory_gb:.2f} GB")
    print("Benchmark completed successfully.")

dist.destroy_process_group()
