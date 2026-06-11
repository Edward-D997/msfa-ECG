import os
import tensorflow as tf
import numpy as np
import sys
import pandas as pd
import tensorflow_addons as tfa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# ---------- 使用 centralized recon 的配置与数据加载 ----------
from utils_centralized_reconv import args, read_tfrecords

# ---------- 导入 MSFA 自定义层（加载模型时需要） ----------
from model import MSFA1D

# ==================== 工具函数（保持不变） ====================
def getcc(ecg12, gen_ecg12, axis=1):
    return (tf.reduce_mean(ecg12 * gen_ecg12, axis=[axis]) -
            tf.reduce_mean(ecg12, axis=[axis]) * tf.reduce_mean(gen_ecg12, axis=[axis])) / \
           (tf.math.reduce_std(ecg12, axis=[axis]) * tf.math.reduce_std(gen_ecg12, axis=[axis]))

def paddingecg(ecg12, index):
    ecg_new = tf.transpose(tf.zeros_like(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg12_t = tf.transpose(ecg12, [0, 2, 1])
    updates = tf.gather_nd(ecg12_t, index)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new


def compute_metric(gen_ecg12, ecg12):
    mae = tf.reduce_sum(tf.reduce_mean(tf.abs(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    mse = tf.reduce_sum(tf.reduce_mean(tf.square(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    cc_item = getcc(ecg12, gen_ecg12, axis=1)
    cc_item_np = cc_item.numpy()
    cc_item_np[np.isnan(cc_item_np)] = 0.0
    cc = tf.reduce_sum(tf.convert_to_tensor(cc_item_np, dtype=tf.float32), axis=0)
    return np.asarray(mae), np.asarray(mse), np.asarray(cc)

def write2excel(MAE_test, MSE_test, CC_test, excel_path):
    def clean_array(arr):
        return np.array(arr.tolist(), dtype=np.float64)
    m_mae, m_mse, m_cc = clean_array(MAE_test), clean_array(MSE_test), clean_array(CC_test)
    def add_mean(arr):
        arr = np.concatenate([arr, np.mean(arr, axis=1)[:, None]], axis=1)
        arr = np.concatenate([arr, np.mean(arr, axis=0)[None]], axis=0)
        return arr
    m_mae, m_mse, m_cc = add_mean(m_mae), add_mean(m_mse), add_mean(m_cc)
    df1 = pd.DataFrame(m_mae, copy=True).round(4)
    df2 = pd.DataFrame(m_mse, copy=True).round(4)
    df3 = pd.DataFrame(m_cc, copy=True).round(4)
    new_excel_path = excel_path.replace(".xlsx", "_final.xlsx")
    with pd.ExcelWriter(new_excel_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='MAE_test')
        df2.to_excel(writer, sheet_name='MSE_test')
        df3.to_excel(writer, sheet_name='CC_test')

# ==================== 评估函数 ====================
def test_ae(model, ds, numlead=12, padding='zeros'):
    MAE_ = np.zeros((numlead, 12))
    MSE_ = np.zeros((numlead, 12))
    CC_ = np.zeros((numlead, 12))
    total_samples = 0
    for step, ecg12 in enumerate(ds):
        print(f"Processing Step: {step}")
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1) < 1e-4)[0], axis=0)
        batch_size = ecg12.shape[0]
        total_samples += batch_size
        l_index = np.arange(batch_size).reshape(-1, 1)
        for i in range(numlead):
            h_index = i * np.ones((batch_size, 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1 = paddingecg(ecg12, index)
            gen_ecg12 = model(ecg1, training=False)
            mae1, mse1, cc1 = compute_metric(gen_ecg12, ecg12)
            MAE_[i, :] += mae1
            MSE_[i, :] += mse1
            CC_[i, :] += cc1
    return MAE_ / total_samples, MSE_ / total_samples, CC_ / total_samples

# ==================== 主程序 ====================
if __name__ == '__main__':
    # ---------- 路径配置 ----------
    testpath = '../datasets/ptbxl_testset'
    # modelpath = '../Ablation_Exp/ablation_MSFA/MSFA_Baseline'
    # ---------- 需要评估的模型目录列表 ----------
    model_dirs = [
        '../Ablation_Exp/ablation_MSFA/MSFA_NoGate',
        '../ablation_MSFA/MSFA_NoSE',
        '../ablation_MSFA/MSFA_SingleScale',
        # 如果有其他模型，继续添加即可
    ]
    resultpath = '../results/Attention_03/MSFA'

    # ---------- 结果保存根目录 ----------
    base_resultpath = '../results/Attention_03/MSFA'

    # ---------- 自定义对象（加载模型时需要） ----------
    custom_objects = {
        'MSFA1D': MSFA1D,
        'GELU': tfa.layers.GELU,
        'InstanceNormalization': tfa.layers.InstanceNormalization,
        'Add': tf.keras.layers.Add,
    }

    # ---------- 循环评估每个模型 ----------
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)  # 例如 MSFA_NoGate
        resultpath = os.path.join(base_resultpath, model_name)
        os.makedirs(resultpath, exist_ok=True)
        excel_path = os.path.join(resultpath, f'{model_name}_ptbxl_testset.xlsx')

        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"Model path: {model_dir}")
        print(f"Results will be saved to: {excel_path}")

        # 加载模型（每个模型独立加载，避免权重冲突）
        model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects, compile=False)

        # 重新构建数据集（保证每个模型都能从起始位置完整遍历）
        testds = read_tfrecords(testpath).batch(args.bs).prefetch(tf.data.experimental.AUTOTUNE)

        # 评估并保存
        MAE_test, MSE_test, CC_test = test_ae(model=model, ds=testds)
        write2excel(MAE_test, MSE_test, CC_test, excel_path)
        print(f"Results saved for {model_name}.\n")

    print("All models evaluated.")