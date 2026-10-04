# -*- coding: utf-8 -*-
"""进程级模型加载锁

torch 的权重反序列化路径不是线程安全的：多个 HuggingFace 模型并发 from_pretrained
会在 modeling_utils.py:748 _load_state_dict_into_meta_model → torch/storage.py:471
__getitem__ 处段错误（exit 139）。

实测：Qwen3-TTS 1.7B 加载耗时约 40 秒，将竞态窗口从毫秒放大到几十秒，
几乎每次重启后第一次并发加载都会撞上。

此锁让所有懒加载的模型串行化。即使 main.py 的 startup 钩子顺序预加载，
这把锁仍然作为兜底——防止未来新增的懒加载点再次撞车。
"""
import threading

#: 进程级共享锁。任何 from_pretrained / torch.load 都必须持有此锁。
MODEL_LOAD_LOCK = threading.Lock()
