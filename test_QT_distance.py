"""
根据 utils_centralized_reconv.py 修改的评估代码（适配 MSFA 模型）。
评估指标：心率(HR)保持能力、QT间期保持能力（配对心搏MAE）。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True)

from utils_centralized_reconv import args, read_tfrecords
# 导入 MSFA 模型构建函数
from model import build_msfa, MSFA1D

import pandas as pd
import neurokit2 as nk

# ========== 补充缺失的 fs 参数（utils_centralized_reconv 未定义） ==========
if not hasattr(args, 'fs'):
    args.fs = 500   # 常用心电采样率，可根据数据集调整

# ========== 信号处理函数 ==========
def extract_qt(ecg, fs, verbose=False):
    """基于 R 峰和局部极值搜索的 QT 间期估计（适用于短信号）"""
    qt_intervals = []
    if np.std(ecg) < 1e-4:
        return []
    try:
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        r_peaks = info.get("ECG_R_Peaks", None)
        if r_peaks is None or len(r_peaks) < 2:
            return []
        r_peaks = np.array(r_peaks)
    except Exception:
        return []

    for r_idx in r_peaks:
        q_window_start = max(0, r_idx - int(0.1 * fs))
        q_window_end = max(0, r_idx - int(0.02 * fs))
        if q_window_end > q_window_start:
            q_segment = ecg_clean[q_window_start:q_window_end]
            if len(q_segment) > 0:
                q_local_min = np.argmin(q_segment)
                q_peak = q_window_start + q_local_min
            else:
                continue
        else:
            continue

        t_window_start = min(len(ecg_clean)-1, r_idx + int(0.15 * fs))
        t_window_end   = min(len(ecg_clean)-1, r_idx + int(0.45 * fs))
        if t_window_end > t_window_start:
            t_segment = ecg_clean[t_window_start:t_window_end]
            if len(t_segment) > 0:
                t_local_max = np.argmax(t_segment)
                t_peak = t_window_start + t_local_max
            else:
                continue
        else:
            continue

        qt_ms = (t_peak - q_peak) / fs * 1000.0
        if 200 <= qt_ms <= 600:
            qt_intervals.append(qt_ms)
    return qt_intervals

def extracthr(ecg):
    """提取心率，增加防御性处理"""
    try:
        if np.std(ecg) < 1e-4:
            return 0.0
        ecg_clean = ecg - np.mean(ecg)
        signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=args.fs)
        r_peaks = info["ECG_R_Peaks"]
        if r_peaks is None or len(r_peaks) < 2:
            return 0.0
        rr_intervals = np.diff(r_peaks) / float(args.fs)
        if rr_intervals.size == 0:
            return 0.0
        heart_rate = 60.0 / np.mean(rr_intervals)
        if 30 <= heart_rate <= 250:
            return heart_rate
        return 0.0
    except Exception:
        return 0.0

def get_hrmetric(hr_matrix):
    """计算导联间统计指标"""
    valid_mask = hr_matrix > 0
    sum_sd = 0.0
    sum_cv = 0.0
    sum_range = 0.0
    valid_sample_count = 0

    for i in range(hr_matrix.shape[0]):
        row_data = hr_matrix[i][valid_mask[i]]
        if len(row_data) >= 2:
            sd = np.std(row_data)
            m = np.mean(row_data)
            sum_sd += sd
            sum_cv += sd / (m + 1e-8)
            sum_range += (np.max(row_data) - np.min(row_data))
            valid_sample_count += 1
        elif len(row_data) == 1:
            valid_sample_count += 1

    if valid_sample_count == 0:
        return np.zeros((4, 1))

    return np.array([[sum_sd], [sum_cv], [sum_range], [float(valid_sample_count)]])

def paddingecg(ecg12, index):
    """与 Trainer.paddingecg 功能一致的独立版本"""
    ecg_new = tf.transpose(tf.zeros_like(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg12_trans = tf.transpose(ecg12, [0, 2, 1])
    updates = tf.gather_nd(ecg12_trans, index)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new

def get_r_peaks(ecg, fs):
    """提取 ECG 信号的 R 峰索引数组"""
    if np.std(ecg) < 1e-4:
        return []
    try:
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        r_peaks = info.get("ECG_R_Peaks", None)
        if r_peaks is None:
            return []
        return list(r_peaks)
    except Exception:
        return []

def get_qt_for_beat(ecg, r_peak, fs):
    """给定 R 峰位置，提取该心搏的 QT 间期(ms)"""
    if np.std(ecg) < 1e-4:
        return None
    try:
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)
    except Exception:
        return None

    q_start = max(0, r_peak - int(0.1 * fs))
    q_end = max(0, r_peak - int(0.02 * fs))
    if q_end <= q_start:
        return None
    q_segment = ecg_clean[q_start:q_end]
    if len(q_segment) == 0:
        return None
    q_local_min = np.argmin(q_segment)
    q_peak = q_start + q_local_min

    t_start = min(len(ecg_clean) - 1, r_peak + int(0.15 * fs))
    t_end = min(len(ecg_clean) - 1, r_peak + int(0.45 * fs))
    if t_end <= t_start:
        return None
    t_segment = ecg_clean[t_start:t_end]
    if len(t_segment) == 0:
        return None
    t_local_max = np.argmax(t_segment)
    t_peak = t_start + t_local_max

    qt_ms = (t_peak - q_peak) / fs * 1000.0
    if 200 <= qt_ms <= 600:
        return qt_ms
    return None

def test_ae_hr(model, ds, numlead=12):
    """
    测试 AE 模型的心率及 QT 间期保持能力（配对心搏计算 MAE）
    """
    rhr_total = np.zeros((4, numlead))
    fhr_total = np.zeros((4, numlead))
    maehr_total = np.zeros((1, numlead))

    rqt_all = [[] for _ in range(12)]
    fqt_all = [[] for _ in range(12)]
    qt_abs_errors = [[] for _ in range(12)]

    for step, ecg12 in enumerate(ds):
        print(f"\nProcessing Step: {step}")
        ecg12_np = ecg12.numpy()
        batch_size = ecg12_np.shape[0]

        # 真实信号心率与R峰提取
        hr1_batch = np.zeros((batch_size, 12))
        r_peaks_all = []
        for b in range(batch_size):
            sample_r_peaks = []
            for l in range(12):
                sig = ecg12_np[b, :, l]
                hr1_batch[b, l] = extracthr(sig)
                r_peaks = get_r_peaks(sig, args.fs)
                sample_r_peaks.append(r_peaks)
                qt_list = extract_qt(sig, args.fs)
                rqt_all[l].extend(qt_list)
            r_peaks_all.append(sample_r_peaks)

        # 模型重建（逐导联）
        padding_len = args.ecglen - ecg12.shape[1] % args.ecglen
        if padding_len == args.ecglen:
            padding_len = 0  # 恰好整除时无需padding
        if padding_len > 0:
            ecg12_padded = tf.concat([ecg12, tf.zeros_like(ecg12)[:, -padding_len:, :]], axis=1)
        else:
            ecg12_padded = ecg12
        ecg12_reshaped = tf.reshape(ecg12_padded, shape=(-1, args.ecglen, 12))
        l_index = np.arange(ecg12_reshaped.shape[0]).reshape(-1, 1)

        gen_ecg12_full_np = np.zeros_like(ecg12_np)

        for i in range(numlead):
            h_index = i * np.ones((ecg12_reshaped.shape[0], 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg_input = paddingecg(ecg12_reshaped, index)

            gen_out = model(ecg_input, training=False)
            # 如果输出为证据分布形式（最后一维48），取前12通道
            if gen_out.shape[-1] == 48:
                gen_out = gen_out[..., :12]
            gen_out = tf.reshape(gen_out, (-1, ecg12.shape[1] + padding_len, 12))
            gen_ecg12_np = gen_out[:, :-padding_len, :].numpy() if padding_len > 0 else gen_out.numpy()

            gen_single_lead = gen_ecg12_np[:, :, i]
            for b in range(batch_size):
                gen_ecg12_full_np[b, :, i] = gen_single_lead[b, :]

            # 计算该导联的心率
            hr2_batch = np.zeros((batch_size, 12))
            for b in range(batch_size):
                for l in range(12):
                    if l <= i:
                        gen_sig = gen_ecg12_full_np[b, :, l]
                        hr2_batch[b, l] = extracthr(gen_sig)
                    else:
                        hr2_batch[b, l] = 0.0

            # QT 统计
            for b in range(batch_size):
                gen_sig = gen_single_lead[b, :]
                qt_list_fake = extract_qt(gen_sig, args.fs)
                fqt_all[i].extend(qt_list_fake)

                real_r_peaks = r_peaks_all[b][i]
                for r_peak in real_r_peaks:
                    qt_real = get_qt_for_beat(ecg12_np[b, :, i], r_peak, args.fs)
                    qt_fake = get_qt_for_beat(gen_sig, r_peak, args.fs)
                    if qt_real is not None and qt_fake is not None:
                        qt_abs_errors[i].append(abs(qt_real - qt_fake))

            if i == numlead - 1:
                mask = (hr1_batch > 0) & (hr2_batch > 0)
                if np.any(mask):
                    maehr_total[0, :] += np.sum(np.abs(hr1_batch - hr2_batch), axis=0)
                rhr_total[:, :] += get_hrmetric(hr1_batch)
                fhr_total[:, :] += get_hrmetric(hr2_batch)

    def calc_qt_stats(qt_global_list):
        stats = np.zeros((3, 12))
        for l in range(12):
            data = qt_global_list[l]
            if len(data) > 0:
                stats[0, l] = np.mean(data)
                stats[1, l] = np.std(data)
                stats[2, l] = len(data)
            else:
                stats[0, l] = np.nan
                stats[1, l] = np.nan
                stats[2, l] = 0
        return stats

    real_qt_stats = calc_qt_stats(rqt_all)
    fake_qt_stats = calc_qt_stats(fqt_all)

    qt_mae_stats = np.zeros((3, 12))
    for l in range(12):
        errors = qt_abs_errors[l]
        if len(errors) > 0:
            qt_mae_stats[0, l] = np.mean(errors)
            qt_mae_stats[1, l] = np.std(errors)
            qt_mae_stats[2, l] = len(errors)
        else:
            qt_mae_stats[0, l] = np.nan
            qt_mae_stats[1, l] = np.nan
            qt_mae_stats[2, l] = 0

    print("\n===== QT Pairwise MAE Summary =====")
    print("QT MAE (ms) per lead:", qt_mae_stats[0, :].round(2))
    print("QT Error STD (ms) per lead:", qt_mae_stats[1, :].round(2))
    print("Paired beats count per lead:", qt_mae_stats[2, :])
    print("===================================\n")

    valid_nums = rhr_total[-1, :]
    safe_valid_nums = np.where(valid_nums == 0, 1e-8, valid_nums)
    final_maehr = maehr_total / (1.0 * safe_valid_nums)
    final_rhr = final_get(rhr_total)
    final_fhr = final_get(fhr_total)

    return final_maehr, final_rhr, final_fhr, real_qt_stats, fake_qt_stats, qt_mae_stats

def final_get(a):
    denom = a[-1, :][None, :]
    denom = np.where(denom == 0, 1e-8, denom)
    return a[:-1, :] / denom

def write2excel_all(rhr, fhr, maehr, rqt, fqt, qt_mae, excel_path):
    df_rhr = pd.DataFrame(rhr, index=["MHR_SD", "MHR_CV", "MHR_Range"]).round(4)
    df_fhr = pd.DataFrame(fhr, index=["MHR_SD", "MHR_CV", "MHR_Range"]).round(4)
    df_mae = pd.DataFrame(maehr, index=["MAE_HeartRate"]).round(4)

    df_rqt = pd.DataFrame(rqt, index=["QT_Mean(ms)", "QT_STD", "Beat_Count"]).round(4)
    df_fqt = pd.DataFrame(fqt, index=["QT_Mean(ms)", "QT_STD", "Beat_Count"]).round(4)

    df_qt_mae = pd.DataFrame(qt_mae, index=["QT_MAE(ms)", "QT_Error_STD", "Paired_Beats"]).round(4)

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_rhr.to_excel(writer, sheet_name='real_heartrate')
        df_fhr.to_excel(writer, sheet_name='fake_heartrate')
        df_mae.to_excel(writer, sheet_name='MAE_HR')
        df_rqt.to_excel(writer, sheet_name='real_QT')
        df_fqt.to_excel(writer, sheet_name='fake_QT')
        df_qt_mae.to_excel(writer, sheet_name='QT_Pairwise_MAE')

if __name__ == '__main__':
    # ---------- 路径配置 ----------
    testpath = '../datasets/cpsc2018'
    resultpath = '../results/Attention_03/MSFA'
    os.makedirs(resultpath, exist_ok=True)

    # ---------- 模型构建（MSFA） ----------
    modelpath = '../Ablation_Exp/ablation_MSFA/MSFA_Baseline'
    # print(f"Loading trained model from {modelpath} ...")

    custom_objects = {
        'MSFA1D': MSFA1D,
        'GELU': tfa.layers.GELU,
        'InstanceNormalization': tfa.layers.InstanceNormalization,
    }

    model = tf.keras.models.load_model(modelpath,
                                       custom_objects=custom_objects,
                                       compile=False)   # 不加载优化器状态，仅用于推理
    print("Model loaded successfully.")
    # model.summary()

    # ---------- 数据加载 ----------
    print("Loading test dataset...")
    testds = read_tfrecords(testpath).batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

    # ---------- 执行评估 ----------
    maehr, rhr, fhr, rqt_stats, fqt_stats, qt_mae_stats = test_ae_hr(model=model, ds=testds)

    excel_path = os.path.join(resultpath, "MSFA_cpsc2018_paired.xlsx")
    write2excel_all(rhr, fhr, maehr, rqt_stats, fqt_stats, qt_mae_stats, excel_path)
    print(f"All results saved to {excel_path}")