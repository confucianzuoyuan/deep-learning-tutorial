首先将之前的一些文件删除.

```bash
$ rm meta.pkl train.bin val.bin ckpt.pt
```

准备数据:

```bash
$ python prepare.py
```

训练模型:

```bash
$ python train.py
```

生成类似三国演义风格的文本:

```bash
$ python sample.py
```