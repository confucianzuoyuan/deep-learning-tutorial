import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# I/O
log_interval = 1
eval_iters = 20
# 数据
gradient_accumulation_steps = 40 # 用来模拟更大的批
batch_size = 12 # 最小的批的大小
block_size = 64
# 模型的配置
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # 在层归一化(LayerNorm)和线性变换(nn.Linear)中不使用偏置.
# AdamW优化器(adamw optimizer)
learning_rate = 1e-3 # 最大学习率
max_iters = 2000 # 训练2000轮
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# 关于学习率衰减的配置.
decay_lr = True # 是否衰减学习率.
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # 最小学习率, 取learning_rate/10.
# system
device = 'cpu' # 使用cpu训练模型
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cpu' # for later use in torch.autocast
ctx = nullcontext()

# 读取训练数据集
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
# 读取测试数据集
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 《三国演义》文本中索引`i`到`i+block_size`的字符串为训练数据的输入数据.
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # 《三国演义》文本中索引`i+1`到`i+1+block_size`的字符串为训练数据的输出数据(标签数据).
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # 将张量移动到CPU上进行计算.
    x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# 读取数据集的元数据.
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']
print(f"字符集的大小是: {meta_vocab_size}")

# 模型初始化的一些超参数. 会覆盖掉GPTConfig的初始配置.
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size,
    dropout=dropout
)

# 从零训练一个新的模型
print("从零训练一个新的模型")
# 配置模型的超参数.
gptconf = GPTConfig(**model_args)
# 实例化一个模型.
model = GPT(gptconf)
# 使用cpu训练模型
model.to(device)

# scaler的作用是使用混合精度的方式训练模型.
# 混合精度的作用: 降低内存使用, 加快训练速度.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 配置AdamW优化器.
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学习率衰减调度器.
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# 训练模型的循环.
X, Y = get_batch('train') # X为训练数据中的输入数据, Y为训练数据中的输出数据(标签数据).
t0 = time.time()
while True:
    # 设置每一轮训练的学习率.
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 计算训练数据集的损失和测试数据集的损失, 并决定是否保存模型的检查点文件.
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"将模型的检查点文件保存到 `ckpt.pt`")
                torch.save(checkpoint, 'ckpt.pt')

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # 将损失(误差)反向传播.
        scaler.scale(loss).backward()
    # 修剪梯度.
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 更新网络中的参数(权重参数和偏置参数).
    scaler.step(optimizer)
    scaler.update()
    # 将梯度从内存中清空, 因为本轮训练的梯度不再需要了.
    optimizer.zero_grad(set_to_none=True)

    # 计算每轮训练的时间长度.
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() # 将损失转换为float类型的浮点数.
        print(f"第{iter_num}轮: 损失大小 {lossf:.4f}, 本轮训练时长 {dt*1000:.2f}ms")
    iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break
