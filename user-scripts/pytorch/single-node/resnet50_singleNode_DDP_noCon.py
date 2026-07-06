import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ------------------------------------------------------------
# Purpose:
# Multi-GPU Distributed Data Parallel (DDP) training benchmark
# using ResNet-50.
#
# This script extends the previous single-GPU workflow into
# a distributed multi-GPU workflow.
#
# IMPORTANT:
# 1 process = 1 GPU
#
# Number of GPUs is controlled during launch:
#
# Example:
# torchrun --nproc_per_node=2 resnet50_ddp.py
#
# To change GPU count:
# --nproc_per_node=4  --> use 4 GPUs
# --nproc_per_node=8  --> use 8 GPUs
#
# This script supports:
# - synthetic dataset
# - train/test split
# - DataLoader
# - DistributedSampler
# - multi-GPU DDP training
# - evaluation
# ------------------------------------------------------------


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
# DDP INITIALIZATION
# ------------------------------------------------------------

# local_rank = GPU ID used by THIS process
local_rank = int(os.environ["LOCAL_RANK"])

# rank = global process ID
rank = int(os.environ["RANK"])

# world_size = total number of processes / GPUs
world_size = int(os.environ["WORLD_SIZE"])

# Initialize distributed communication backend
dist.init_process_group(backend="nccl")

# Bind this process to its GPU
torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}")

# Print ONLY from main process
if rank == 0:
    print("===== Multi-GPU ResNet-50 DDP Benchmark =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"World size (total GPUs): {world_size}")


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

dataset_size = 2000000
train_ratio = 0.8
num_classes = 10
image_size = 224

# IMPORTANT:
# This is PER-GPU batch size
#
# Example:
# batch_size = 256
# world_size = 2 GPUs
#
# Effective total batch size:
# 256 x 2 = 512
batch_size = 1024

epochs = 20
num_workers = 4

if rank == 0:
    print(f"Dataset size: {dataset_size}")
    print(f"Train/test split: {int(train_ratio * 100)}% / {int((1 - train_ratio) * 100)}%")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {image_size} x {image_size}")
    print(f"Per-GPU batch size: {batch_size}")
    print(f"Effective total batch size: {batch_size * world_size}")
    print(f"Epochs: {epochs}")
    print(f"DataLoader workers: {num_workers}")


# ------------------------------------------------------------
# PERFORMANCE SETTINGS
# ------------------------------------------------------------

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

dataset = SyntheticImageDataset(
    dataset_size,
    num_classes,
    image_size
)

train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# ------------------------------------------------------------
# DISTRIBUTED SAMPLERS
# ------------------------------------------------------------
# VERY IMPORTANT:
# Each GPU/process receives DIFFERENT subset of data
# ------------------------------------------------------------

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

test_sampler = DistributedSampler(
    test_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False
)

# ------------------------------------------------------------
# DATALOADERS
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

# ------------------------------------------------------------
# DDP WRAPPER
# ------------------------------------------------------------
# This synchronizes gradients across GPUs
# ------------------------------------------------------------

model = DDP(
    model,
    device_ids=[local_rank]
)

model.train()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

torch.cuda.reset_peak_memory_stats()

total_train_samples = 0

train_start_time = time.time()

# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------

if rank == 0:
    print("===== Training =====")

for epoch in range(epochs):

    # IMPORTANT:
    # ensures proper shuffling across epochs in DDP
    train_sampler.set_epoch(epoch)

    model.train()

    epoch_start_time = time.time()

    running_loss = 0.0
    epoch_samples = 0

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
        epoch_samples += batch_samples
        total_train_samples += batch_samples

    torch.cuda.synchronize()

    epoch_end_time = time.time()

    epoch_time = epoch_end_time - epoch_start_time

    epoch_loss = running_loss / epoch_samples

    epoch_throughput = epoch_samples / epoch_time

    # Print ONLY from main process
    if rank == 0:
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train loss: {epoch_loss:.4f}")
        print(f"  Epoch time: {epoch_time:.4f} seconds")
        print(f"  Per-process throughput: {epoch_throughput:.2f} samples/sec")
        print(f"  Estimated total throughput: {epoch_throughput * world_size:.2f} samples/sec")


torch.cuda.synchronize()

train_end_time = time.time()

total_train_time = train_end_time - train_start_time

overall_train_throughput = total_train_samples / total_train_time


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

if rank == 0:
    print("===== Evaluation =====")

model.eval()

test_loss = 0.0
correct = 0
total_test_samples = 0

eval_start_time = time.time()

with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        batch_samples = inputs.size(0)

        test_loss += loss.item() * batch_samples

        total_test_samples += batch_samples

        predicted = outputs.argmax(dim=1)

        correct += (predicted == targets).sum().item()


torch.cuda.synchronize()

eval_end_time = time.time()

eval_time = eval_end_time - eval_start_time

avg_test_loss = test_loss / total_test_samples

test_accuracy = correct / total_test_samples

eval_throughput = total_test_samples / eval_time

peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3


# ------------------------------------------------------------
# FINAL RESULTS
# ------------------------------------------------------------

if rank == 0:

    print("===== Final Results =====")

    print(f"Total training time: {total_train_time:.4f} seconds")

    print(f"Per-process throughput: {overall_train_throughput:.2f} samples/sec")

    print(f"Estimated total throughput: {overall_train_throughput * world_size:.2f} samples/sec")

    print(f"Evaluation time: {eval_time:.4f} seconds")

    print(f"Evaluation throughput: {eval_throughput:.2f} samples/sec")

    print(f"Final test loss: {avg_test_loss:.4f}")

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    print(f"Peak GPU memory allocated (per GPU): {peak_memory_gb:.2f} GB")

    print("Benchmark completed successfully.")


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

dist.destroy_process_group()
