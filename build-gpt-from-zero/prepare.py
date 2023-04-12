import pickle
import numpy as np

# 打开《三国演义》文本文件, 然后读取到`data`.
with open('input.txt', 'r') as f:
    data = f.read()
print(f"数据集中的字符数量: {len(data):,}")

# 获取《三国演义》中的所有不同的字符.
# 对数据使用`set`去重, 然后转换成列表并排序.
chars = sorted(list(set(data)))
# 字符集的数量.
vocab_size = len(chars)
# 打印所有字符.
print("所有不同字符:", ''.join(chars))
# 打印字符集大小.
print(f"字符集大小: {vocab_size:,}")

# 创建字符到整数的映射字典
stoi = { ch:i for i,ch in enumerate(chars) }
# 创建整数到字符的映射字典
itos = { i:ch for i,ch in enumerate(chars) }
# 将字符串编码为整数列表.
# encode('滚滚长江东逝水') --> [2044, 2044, 3600, 1881, 40, 3429, 1871]
def encode(s):
    return [stoi[c] for c in s]
# 将整数列表解码为字符串.
# decode([2044, 2044, 3600, 1881, 40, 3429, 1871]) --> '滚滚长江东逝水'
def decode(l):
    return ''.join([itos[i] for i in l])

# 将《三国演义》文本的前90%作为训练数据
# 将《三国演义》文本的后10%作为测试数据
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 将训练数据集和测试数据集都编码为整数列表.
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"训练数据集有 {len(train_ids):,} tokens")
print(f"测试数据集有 {len(val_ids):,} tokens")

# 将数据集保存到磁盘.
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

# 保存数据集的元数据.
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
