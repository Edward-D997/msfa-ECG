import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import h5py
import math
import argparse
from tensorflow.keras.utils import Sequence
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score,
                             precision_recall_curve)

# 自定义层，用于加载重建模型
from model_MSFA_ablation import MSFA1D, MSFA1D_NoGate, MSFA1D_NoSE


# ---------- 数据序列（未改动核心逻辑） ----------
class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()

# ---------- 工具函数 ----------
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T

def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc

def paddingecg(ecg12, index):
    ecg12 = tf.transpose(tf.cast(ecg12, dtype=tf.float32), [0, 2, 1])
    updates = tf.gather_nd(ecg12, index)
    ecg_new = tf.zeros_like(ecg12, dtype=tf.float32)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new

def updateecg(ecg12, ecg1, index):
    ecg12 = tf.transpose(tf.cast(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg1 = tf.transpose(tf.cast(ecg1, dtype=tf.float32), [0, 2, 1])
    updates = tf.gather_nd(ecg1, index)
    ecg12 = tf.tensor_scatter_nd_update(ecg12, index, updates)
    ecg12 = tf.transpose(tf.cast(ecg12, dtype=tf.float32), [0, 2, 1])
    return ecg12

def convert_df(y, y_true, threshold=np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])):
    y = (y > threshold).astype(np.float32)
    scores = get_scores(y_true, y, score_fun)
    scores = np.concatenate([scores, np.mean(scores, axis=0)[None, :]], axis=0)
    df = pd.DataFrame(scores,
                      columns=['Precision', 'Recall', 'Specificity', 'F1 score', 'Acc'],
                      index=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'Mean'])
    return df

# ---------- 评估单个重建模型 ----------
def evaluate_reconstruction_model(generator, ecg12_data, y_true, class_model, score_fun, threshold, lead_names):
    """
    对给定重建模型 generator，使用 12 个单导联输入重建并分类，
    返回合并后的多级列 DataFrame（行=导联，列=(诊断类别, 指标)）。
    """
    ecg12_reshaped = tf.reshape(ecg12_data, shape=(-1, 1024, 12))  # 适配重建模型
    num_samples = ecg12_reshaped.shape[0]
    
    results = []   # 列表元素: (导联索引, DataFrame)
    for i in range(12):
        l_index = np.arange(num_samples).reshape(-1, 1)
        h_index = i * np.ones((num_samples, 1)).astype(np.int32)
        index = np.hstack((l_index, h_index))

        ecg1 = paddingecg(ecg12_reshaped, index)
        gen_ecg12 = generator.predict(ecg1, verbose=0)
        gen_ecg12 = gen_ecg12[:, :, :12]          # 确保取到重建的12导联
        gen_ecg12 = updateecg(gen_ecg12, ecg1, index)  # 保持输入导联为真值

        # 恢复为原始长度 (4096, 12) 以匹配分类模型
        gen_ecg12_reshaped = tf.reshape(gen_ecg12, shape=(-1, 4096, 12))
        y_pred_prob = class_model.predict(gen_ecg12_reshaped, verbose=0)
        df_lead = convert_df(y_pred_prob, y_true, threshold)
        results.append(df_lead)

    # 将 12 个 DataFrame 合并为多级列
    # 每个 df_lead: index=诊断类别, columns=指标
    # 合并后行索引为 lead_names, 列索引为 MultiIndex (诊断类别, 指标)
    merged_dict = {}
    for lead_name, df_lead in zip(lead_names, results):
        for diag in df_lead.index:
            for metric in df_lead.columns:
                merged_dict[(lead_name, diag, metric)] = df_lead.loc[diag, metric]
    
    # 转换为 Series 再 unstack
    multi_index = pd.MultiIndex.from_tuples(merged_dict.keys(), names=['Lead', 'Diagnosis', 'Metric'])
    s = pd.Series(list(merged_dict.values()), index=multi_index)
    df_merged = s.unstack(level=[1, 2])  # 行: Lead, 列: MultiIndex (Diagnosis, Metric)
    # 按正常导联顺序排序
    df_merged = df_merged.reindex(lead_names)
    return df_merged

# ==================== 主程序 ====================
if __name__ == '__main__':
    # 阈值设定
    threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    score_fun = {'Precision': precision_score,
                 'Recall': recall_score,
                 'Specificity': specificity_score,
                 'F1 score': f1_score,
                 'acc': accuracy_score}

    # 路径配置
    path_to_hdf5 = '../datasets/code-test/data/ecg_tracings.hdf5'
    dataset_name = 'tracings'
    labelpath = '../datasets/code-test/data/annotations/gold_standard.csv'
    class_model_path = '../datasets/code-test/data/model.hdf5'

    # 加载金标准
    y_true = pd.read_csv(labelpath).values

    # 加载 ECG 数据
    ecgdata = h5py.File(path_to_hdf5, 'r')
    ecg12_all = ecgdata[dataset_name]  # 原始长信号 (N, 4096, 12)

    # 加载分类模型
    class_model = tf.keras.models.load_model(class_model_path, compile=False)
    class_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())

    # 原始12导联分类结果（作为基准）
    print("Evaluating original 12-lead ECG...")
    yo = class_model.predict(ecg12_all, verbose=1)
    df_original = convert_df(yo, y_true, threshold)

    # ---------- 定义需要评估的重建模型列表 ----------
    # 每个字典包含：路径（path），名称（name），加载时所需的 custom_objects: MSFA1D, MSFA1D_NoGate, MSFA1D_NoSE
    reconstruction_models = [
        {
            'path': '../Ablation_Exp/ablation_MSFA/MSFA_NoGate',
            'name': 'MSFA_MSFA_NoGate',
            'custom_objects': {
                'GELU': tfa.layers.GELU,
                'MSFA1D_NoGate': MSFA1D_NoGate,
                'InstanceNormalization': tfa.layers.InstanceNormalization,
                'LayerNormalization': tf.keras.layers.LayerNormalization,
            }
        },
        {
            'path': '../Ablation_Exp/ablation_MSFA/MSFA_NoSE',
            'name': 'MSFA_NoSE',
            'custom_objects': {
                'MSFA1D_NoSE': MSFA1D_NoSE,
                'GELU': tfa.layers.GELU,
                'InstanceNormalization': tfa.layers.InstanceNormalization,
                'LayerNormalization': tf.keras.layers.LayerNormalization,
            }
        },
        # 你可以继续添加其他模型，例如：
        # {
        #     'path': '/path/to/RevConv_FullAttn_DTA',
        #     'name': 'RevConv_FullAttn_DTA',
        #     'custom_objects': { ... }  # 注意 DTA 可能不需要额外层，但 ReverseConv1D 仍需
        # },
    ]

    # 结果输出目录
    output_dir = '../results/Attention_03/ablation_no_reconv'
    os.makedirs(output_dir, exist_ok=True)

    # 逐个评估重建模型
    for model_cfg in reconstruction_models:
        print(f"\n{'='*60}")
        print(f"Evaluating reconstruction model: {model_cfg['name']}")
        print(f"{'='*60}")

        # 加载重建模型
        generator = tf.keras.models.load_model(
            model_cfg['path'],
            custom_objects=model_cfg['custom_objects'],
            compile=False
        )

        # 执行评估
        df_recon = evaluate_reconstruction_model(
            generator, ecg12_all, y_true, class_model, score_fun, threshold, lead_names
        )

        # 写入 Excel：一个文件，两个 sheet
        excel_path = os.path.join(output_dir, f"{model_cfg['name']}_classification.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_recon.to_excel(writer, sheet_name='Reconstructed')
            df_original.to_excel(writer, sheet_name='Original_12Lead')
        print(f"Results saved to {excel_path}")

    ecgdata.close()
    print("\nAll evaluations completed.")