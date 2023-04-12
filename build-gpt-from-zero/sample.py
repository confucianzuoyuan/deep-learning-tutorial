import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
start = "曹操" # prompt提示文本, 相当于chapGPT中的提问.
num_samples = 10 # 以`start`开头写几段文字.
max_new_tokens = 500 # 每段文字多少个token
temperature = 0.8 # 调节随机程度
top_k = 200 # 保留`top_k`个最可能的token
seed = 1337 # 随机数种子
device = 'cpu' # 推理设备使用`cpu`
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cpu' # for later use in torch.autocast
ctx = nullcontext()

# 加载模型的检查点文件.
checkpoint = torch.load('ckpt.pt', map_location=device)
# 读取模型的配置.
gptconf = GPTConfig(**checkpoint['model_args'])
# 实例化模型.
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

print(f"从 `meta.pkl` 文件加载元数据")
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 将prompt提示文本编码为整数列表.
start_ids = encode(start)
# 将整数列表转换成张量.
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 生成文本.
# 禁用梯度计算.
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # 根据x预测得到y
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # 将预测得到的y解码为文本字符串.
            print(decode(y[0].tolist()))
            print('---------------')
