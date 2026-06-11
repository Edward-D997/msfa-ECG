# run_MSFA_ablation.py
import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 MSFA 系列模型
from model_MSFA_ablation import (
    model_msfa_single_scale,      # single scale，验证多尺度交互：SingleScale 退化导致性能大幅下降
    model_msfa_no_gate,           # 去掉门控，验证预测‑校正机制：使融合变为简单拼接，性能受损
    model_msfa_no_se,           # 去掉通道注意力：会引入少量性能损失，说明自适应筛选校正信息是必要的
)
from utils_centralized_reconv import Trainer, mse_loss, args, read_tfrecords

# ---------- 路径与超参 ----------
TRAIN_PATH = '../datasets/ptbxl_trainset'
VAL_PATH = '../datasets/ptbxl_valset'
BASE_SAVE_DIR = '../Ablation_Exp/ablation_MSFA'
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

trainds = read_tfrecords(TRAIN_PATH).shuffle(1000).batch(args.bs).prefetch(tf.data.AUTOTUNE)
valds = read_tfrecords(VAL_PATH).batch(args.bs).prefetch(tf.data.AUTOTUNE)

STEP_TOTAL = 87200 // args.bs
EPOCHS = 150
PATIENCE = 30
LR = 1e-3

# ---------- MSFA 消融实验配置 ----------
ablation_configs = [
    # MCMA（基线) => MSFA 基线（纯跳跃连接替换，无额外模块）:证明新方法的提升
    # SingleScale / NoGate / NoSE => 证明各组件的作用
    # {
    #     'exp_name': 'MSFA_SingleScale',
    #     'model_fn': model_msfa_single_scale,
    #     'loss_fn': mse_loss,
    #     'loss_kwargs': {},
    # },
    # {
    #     'exp_name': 'MSFA_NoGate',
    #     'model_fn': model_msfa_no_gate,
    #     'loss_fn': mse_loss,
    #     'loss_kwargs': {},
    # },
    {
        'exp_name': 'MSFA_NoSE',
        'model_fn': model_msfa_no_se,
        'loss_fn': mse_loss,
        'loss_kwargs': {},
    },
]

# ---------- 顺序训练 ----------
for cfg in ablation_configs:
    print(f"\n{'='*60}")
    print(f"Running: {cfg['exp_name']}")
    print(f"{'='*60}")

    model_dir = os.path.join(BASE_SAVE_DIR, cfg['exp_name'])
    os.makedirs(model_dir, exist_ok=True)

    trainer = Trainer(
        modelpath=model_dir,
        model_fn=cfg['model_fn'],
        loss_fn=cfg['loss_fn'],
        loss_kwargs=cfg['loss_kwargs'],
        epochs=EPOCHS,
        lr=LR,
        ecglen=args.ecglen,
        step_total=STEP_TOTAL,
        anylead=1,
        patience=PATIENCE,
        exp_name=cfg['exp_name']
    )

    trainer.train(trainds, valds)
    