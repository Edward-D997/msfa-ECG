# utils_msfa.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

# ---------- GPU 配置 ----------
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)

strategy = tf.distribute.MirroredStrategy()

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description='MSFA ECG Reconstruction')
parser.add_argument('--ecglen', default=1024, type=int, help='HeartBeat length')
parser.add_argument('--ecglen_Long', default=5000, type=int, help='Recording length (for long records)')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--anylead', default=1, type=int, help='Use any single lead (1) or fixed lead (0)')
parser.add_argument('--lambda_cal', default=0.1, type=float, help='calibration penalty weight (for old CUP)')
parser.add_argument('--lambda_mse', default=0.3, type=float, help='weight of MSE term in hybrid loss')
parser.add_argument('--lambda_rank', default=0.01, type=float, help='weight of rank calibration loss')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--patience', default=50, type=int, help='early stopping patience')

args = parser.parse_args()

# ---------- TFRecord 辅助函数 ----------
def datatorecord4c(tfrecordwriter, ecgs):
    writer = tf.io.TFRecordWriter(tfrecordwriter)
    for i in range(ecgs.shape[0]):
        ecg = ecgs[i]
        ecg = np.asarray(ecg).astype(np.float32).tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'ecg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ecg]))}
            ))
        writer.write(example.SerializeToString())
    writer.close()

def decode_tfrecords4c(example):
    feature_description = {'ecg': tf.io.FixedLenFeature([], tf.string)}
    feature_dict = tf.io.parse_single_example(example, feature_description)
    ecg = tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32)
    ecg = tf.reshape(ecg, [args.ecglen, 12])
    return ecg

def read_tfrecords(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(decode_tfrecords4c, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def decode_tfrecords_Long(example):
    feature_description = {'ecg': tf.io.FixedLenFeature([], tf.string)}
    feature_dict = tf.io.parse_single_example(example, feature_description)
    ecg = tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32)
    ecg = tf.reshape(ecg, [args.ecglen_Long, 12])
    return ecg

def read_tfrecords_Long(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(decode_tfrecords_Long, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def extract(a, t, x_shape):
    batch_size, sequence_length, _ = a.shape
    t_shape = tf.shape(t)
    out = tf.gather(a, t, axis=-1)
    out = tf.reshape(out, (batch_size, t_shape[0], *((1,) * (len(x_shape) - 1))))
    return out

# ========== Trainer 类 ==========
class Trainer:
    def __init__(self, modelpath, model_fn, loss_fn, loss_kwargs=None,
                 epochs=200, lr=1e-3, ecglen=1024,
                 figure_plot=0, step_total=340, anylead=1, patience=50,
                 exp_name='default'):
        self.ecglen = ecglen
        self.epochs = epochs
        self.lr = lr
        self.figure_plot = figure_plot
        self.anylead = anylead
        self.step_total = step_total
        self.patience = patience
        self.modelpath = modelpath
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs or {}
        self.exp_name = exp_name

        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.val_cc_tracker = tf.keras.metrics.Mean(name='val_cc')

        self.model = model_fn(input_size=(self.ecglen, 12))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_tracker = tf.keras.metrics.Mean()

        self.train_cc = []
        self.train_loss = []
        self.val_loss = []
        self.val_cc = []
        self.best_loss1 = np.inf
        self.waiting = 0
        self.eplison = 1e-5

    def getcc(self, ecg12, gen_ecg12):
        numerator = (tf.reduce_mean(ecg12 * gen_ecg12, axis=[1]) -
                     tf.reduce_mean(ecg12, axis=[1]) * tf.reduce_mean(gen_ecg12, axis=[1]))
        denominator = (tf.math.reduce_std(ecg12, axis=[1]) *
                       tf.math.reduce_std(gen_ecg12, axis=[1])) + self.eplison
        return numerator / denominator

    def save_history(self, filename=None):
        if filename is None:
            filename = os.path.join(self.modelpath, f'{self.exp_name}_training_log.csv')
        try:
            data = {
                'epoch': list(range(1, len(self.train_loss) + 1)),
                'train_loss': self.train_loss,
                'train_cc': self.train_cc,
                'val_loss': self.val_loss,
                'val_cc': self.val_cc
            }
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\n[Success] Training history saved to {filename}")
        except Exception as e:
            print(f"\n[Error] Failed to save CSV: {e}")

    @tf.function
    def paddingecg(self, ecg12, index):
        ecg12_trans = tf.transpose(ecg12, [0, 2, 1])
        updates = tf.gather_nd(ecg12_trans, index)
        ecg_new = tf.zeros_like(ecg12_trans, dtype=tf.float32)
        ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
        return tf.transpose(ecg_new, [0, 2, 1])

    @tf.function
    def train_step(self, ecg12):
        def step_fn(ecg12):
            l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
            if self.anylead == 1:
                h_index = np.random.randint(0, 12, ecg12.shape[0]).reshape(-1, 1).astype(np.int32)
            else:
                h_index = np.zeros((ecg12.shape[0], 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1 = self.paddingecg(ecg12, index)

            with tf.GradientTape() as tape:
                output = self.model(ecg1, training=True)
                if output.shape[-1] == 48:
                    loss = self.loss_fn(ecg12, output, **self.loss_kwargs)
                    gamma = output[..., :12]
                else:
                    loss = self.loss_fn(ecg12, output, **self.loss_kwargs)
                    gamma = output

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            cc = self.getcc(ecg12, gamma)
            return loss, tf.reduce_mean(cc)
        return step_fn(ecg12)

    
    @tf.function
    def test_step_comprehensive(self, ecg12):
        batch_size = tf.shape(ecg12)[0]
        total_loss = 0.0
        total_cc = 0.0

        for i in range(12):
            l_index = tf.range(batch_size)[:, tf.newaxis]
            h_index = tf.fill([batch_size, 1], i)
            index = tf.concat([tf.cast(l_index, tf.int32), tf.cast(h_index, tf.int32)], axis=1)

            ecg_input = self.paddingecg(ecg12, index)
            output = self.model(ecg_input, training=False)

            # 统一使用 MSE 作为验证损失（与消融训练损失一致）
            if output.shape[-1] == 48:
                gamma = output[..., :12]
            else:
                gamma = output
            loss = tf.reduce_mean(tf.square(gamma - ecg12))
            total_loss += loss
            cc_val = self.getcc(ecg12, gamma)
            total_cc += tf.reduce_mean(cc_val)

        return total_loss / 12.0, total_cc / 12.0

    def train(self, train_data, val_data):
        steps_per_epoch = self.step_total

        for epoch in range(self.epochs):
            self.loss_tracker.reset_states()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            progbar = tf.keras.utils.Progbar(
                target=steps_per_epoch,
                stateful_metrics=['loss', 'cc', 'val_loss', 'val_cc']
            )
            train_loss_list, train_cc_list = [], []

            for train_batch, ecg12 in enumerate(train_data):
                if train_batch >= steps_per_epoch:
                    break

                t_loss, t_cc = self.train_step(ecg12)
                train_loss_list.append(t_loss)
                train_cc_list.append(t_cc)

                progbar.update(train_batch, values=[('loss', t_loss), ('cc', t_cc)], finalize=False)

            avg_train_loss = np.mean(train_loss_list)
            avg_train_cc = np.mean(train_cc_list)
            self.train_loss.append(avg_train_loss)
            self.train_cc.append(avg_train_cc)

            # 验证
            val_loss_sum, val_cc_sum, val_steps = 0.0, 0.0, 0
            for ecg12 in val_data:
                v_loss, v_cc = self.test_step_comprehensive(ecg12)
                val_loss_sum += v_loss
                val_cc_sum += v_cc
                val_steps += 1

            avg_val_loss = (val_loss_sum / val_steps).numpy()
            avg_val_cc = (val_cc_sum / val_steps).numpy()
            self.val_loss.append(avg_val_loss)
            self.val_cc.append(avg_val_cc)

            progbar.update(steps_per_epoch, values=[
                ('loss', avg_train_loss),
                ('cc', avg_train_cc),
                ('val_loss', avg_val_loss),
                ('val_cc', avg_val_cc)
            ], finalize=True)

            if avg_val_loss < self.best_loss1:
                self.best_loss1 = avg_val_loss
                self.model.save(self.modelpath)
                print(f"  [model saved!] Best Weights saved to {self.modelpath}")
                self.waiting = 0
            else:
                self.waiting += 1

            if self.waiting > self.patience:
                print("Early stopping triggered.")
                break

        self.save_history()

    def sample(self, ecg12, epoch, no_of=1):
        l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
        h_index = np.zeros((ecg12.shape[0], 1)).astype(np.int32)
        index = np.hstack((l_index, h_index))
        ecg1 = self.paddingecg(ecg12, index)
        evi_out = self.model(ecg1, training=False)
        gen_ecg12 = evi_out[..., :12]
        # 绘图代码略
        pass

    def evaluate(self, val_data, use_main=False):
        self.val_loss_tracker.reset_states()
        # 测试代码略，建议使用 test_step_comprehensive
        return self.val_loss_tracker.result()
    
