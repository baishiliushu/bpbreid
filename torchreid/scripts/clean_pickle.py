#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np

def to_builtin(obj):
    """递归清洗 numpy scalar / ndarray，不动 torch.Tensor"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_builtin(v) for v in obj)
    return obj


def clean_checkpoint(input_path, output_path):
    print(f"[clean] loading: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    clean_ckpt = {}

    for k, v in ckpt.items():
        if k == "state_dict":
            # 权重：原样保留
            clean_ckpt[k] = v
        else:
            # 其他字段：清洗 numpy
            clean_ckpt[k] = to_builtin(v)

    # epoch 再保险一次
    if "epoch" in clean_ckpt:
        clean_ckpt["epoch"] = int(clean_ckpt["epoch"])

    print(f"[clean] saving: {output_path}")
    torch.save(clean_ckpt, output_path)
    print("[clean] done")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_checkpoint.py <input.pth.tar> <output.pth.tar>")
        sys.exit(1)

    clean_checkpoint(sys.argv[1], sys.argv[2])
