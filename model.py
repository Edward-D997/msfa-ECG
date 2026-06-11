# MSFA:多尺度融合适配器for 12-lead ECG Reconstruction
import tensorflow as tf
import tensorflow_addons as tfa

# ============================================================
# 多尺度融合适配器 (MSFA1D)
# ============================================================
class MSFA1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.reduction = reduction

    def build(self, input_shape):
        num_x = len(input_shape) - 1

        self.bottleneck = tf.keras.layers.Conv1D(
            self.filters * 2, 1, padding='same', activation='gelu', name='bottleneck')
        self.conv_corr = tf.keras.layers.Conv1D(
            self.filters, self.kernel_size, padding='same', activation='gelu', name='conv_corr')
        self.conv_pred = tf.keras.layers.Conv1D(
            self.filters, self.kernel_size, padding='same', activation='gelu', name='conv_pred')
        self.gate_dense = tf.keras.layers.Dense(self.filters, activation='sigmoid', name='gate_dense')
        self.se_dense1 = tf.keras.layers.Dense(self.filters // self.reduction, activation='relu', name='se1')
        self.se_dense2 = tf.keras.layers.Dense(self.filters, activation='sigmoid', name='se2')
        self.conv_out = tf.keras.layers.Conv1D(self.filters, 1, padding='same', activation='gelu', name='out')

        self.x_projections = []
        for i in range(num_x):
            in_ch = input_shape[i+1][-1]
            if in_ch != self.filters:
                proj = tf.keras.layers.Conv1D(self.filters, 1, padding='same',
                                              name=f'proj_{i}')
            else:
                proj = None
            self.x_projections.append(proj)

        super().build(input_shape)

    def _align_to_y(self, x, target_len):
        x_4d = tf.expand_dims(x, axis=-1)
        x_resized = tf.image.resize(x_4d, size=(target_len, x.shape[-1]), method='bilinear')
        return x_resized[..., 0]

    def call(self, inputs):
        y = inputs[0]
        x_list = inputs[1:]
        y_shape = tf.shape(y)
        target_len = y_shape[1]

        x_projected = []
        for i, x in enumerate(x_list):
            x_aligned = self._align_to_y(x, target_len)
            proj = self.x_projections[i]
            if proj is not None:
                x_aligned = proj(x_aligned)
            x_projected.append(x_aligned)

        x_global = x_projected[0]
        x_local  = x_projected[-1]

        gate_input = tf.concat([x_global, x_local, y], axis=-1)
        gate = self.gate_dense(gate_input)
        y_pred = self.conv_pred(y * gate + x_global * (1 - gate))

        concat = tf.concat([y_pred] + x_projected, axis=-1)
        bottle = self.bottleneck(concat)
        y_corr = self.conv_corr(bottle)

        avg = tf.reduce_mean(y_corr, axis=1, keepdims=True)
        max_ = tf.reduce_max(y_corr, axis=1, keepdims=True)
        se = self.se_dense1(avg) + self.se_dense1(max_)
        se = self.se_dense2(se)

        refined = y + y_corr * se
        return self.conv_out(refined)


# ============================================================
# 基础卷积块
# ============================================================
def downblock(x0, filters, kernel_size=5, strides=1, padding='same'):
    x1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                activation='gelu', padding=padding)(x0)
    x2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                activation='linear', padding=padding)(x0)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tfa.layers.GELU()(x2)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
                                activation='gelu', padding=padding)(x2)
    return tfa.layers.InstanceNormalization(epsilon=1e-9)(x1 + x)


def upblock(x0, filters, kernel_size=5, strides=1, padding='same'):
    x1 = tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                         activation='gelu', padding=padding)(x0)
    x2 = tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                         activation='linear', padding=padding)(x0)
    x2 = tf.keras.layers.LayerNormalization()(x2)
    x2 = tfa.layers.GELU()(x2)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
                                activation='gelu', padding=padding)(x2)
    return tfa.layers.InstanceNormalization(epsilon=1e-9)(x1 + x)


# ============================================================
# MSFA_Base 构建器（固定无额外模块）
# ============================================================
def build_msfa(input_size=(1024, 12), output_channels=12, pool_size=2):
    filters = [16, 32, 64, 128, 256, 512]
    input_layer = tf.keras.layers.Input(shape=input_size, name="input_layer")

    # ---------- Encoder ----------
    e1 = downblock(input_layer, filters[0])
    e2 = downblock(e1, filters[1], strides=pool_size)
    e3 = downblock(e2, filters[2], strides=pool_size)
    e4 = downblock(e3, filters[3], strides=pool_size)
    e5 = downblock(e4, filters[4], strides=pool_size)
    e6 = downblock(e5, filters[5], strides=pool_size)

    # ---------- Decoder with MSFA ----------
    d5_up = upblock(e6, filters[4], strides=pool_size) # 即d6
    d5 = MSFA1D(filters[4], name='MSFA_d5')([d5_up, e6, e5, e4])

    d4_up = upblock(d5, filters[3], strides=pool_size)
    d4_msfa = MSFA1D(filters[3], name='MSFA_d4')([d4_up, e5, e4, e3])
    d4 = downblock(d4_msfa, filters[3])

    d3_up = upblock(d4, filters[2], strides=pool_size)
    d3_msfa = MSFA1D(filters[2], name='MSFA_d3')([d3_up, e4, e3, e2])
    d3 = downblock(d3_msfa, filters[2])

    d2_up = upblock(d3, filters[1], strides=pool_size)
    d2 = MSFA1D(filters[1], name='MSFA_d2')([d2_up, e3, e2, e1])

    d1_up = upblock(d2, filters[0], strides=pool_size)
    d1 = MSFA1D(filters[0], name='MSFA_d1')([d1_up, e2, e1])

    out = tf.keras.layers.Conv1D(output_channels, kernel_size=1, padding='same')(d1)
    return tf.keras.Model(inputs=input_layer, outputs=out, name='MSFA_Base')


# ============================================================
# 模型函数
# ============================================================
def model_msfa_baseline(input_size=(1024, 12)):
    return build_msfa(input_size)


if __name__ == '__main__':
    model = model_msfa_baseline((1024, 12))
    model.summary()