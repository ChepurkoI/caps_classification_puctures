import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, layers


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]

        Вычисляем длину векторов. Это используется для вычисления тензора, который имеет ту же форму, что и y_true в margin_loss.
    Используя этот слой в качестве выхода модели, можно напрямую предсказывать метки с помощью `y_pred = np.argmax(model.predict(x), 1)`.
    входы: shape=[None, num_vectors, dim_vector]
    выход: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
        Маска тензора с shape=[None, num_capsule, dim_vector] либо по капсуле с максимальной длиной, либо по
    дополнительной входной маской. Кроме капсулы с максимальной длиной (или указанной капсулы), все векторы
    маскируются нулями. Затем нормируем тензор с маской.
    Например:
        ```
        x = keras.layers.Input(shape=[8, 3, 2]) # batch_size=8, каждая выборка содержит 3 капсулы с dim_vector=2
        y = keras.layers.Input(shape=[8, 3]) # Истинные метки. 8 образцов, 3 класса, однократное кодирование.
        out = Mask()(x) # out.shape=[8, 6]
        # или
        out2 = Mask()([x, y]) # out2.shape=[8,6]. Маскируются истинными метками y.
        ```
    """

    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # истинная метка предоставляется с shape = [None, n_classes], т.е. one-hot код.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # если нет истинной метки, маска по максимальной длине капсул. В основном используется для предсказания
            # вычислить длину капсул
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            # генерируем маску, которая является one-hot кодом.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])

        masked = K.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # истинная метка предоставлена
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    Нелинейная активация, используемая в Capsule. Она приводит длину большого вектора к значению, близкому к 1, а малого - к 0.
    :param vectors: некоторые векторы для нормировки, N-мерный тензор
    :param axis: ось для нормировки
    :return: тензор с той же формой, что и входные векторы
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    Капсульный слой. Он аналогичен полносвязному слою. Полносвязный слой имеет `in_num` входов, каждый из которых
    является скаляром, выходом нейрона из предыдущего слоя, и имеет `out_num` выходных нейронов.
    CapsuleLayer просто расширяет выход нейрона из скаляра в вектор. Поэтому его
    входная форма = [None, input_num_capsule, input_dim_capsule] и
    выходная форма = [None, num_capsule, dim_capsule].
    Для полносвязного слоя, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: количество капсул в данном слое
    :param dim_capsule: размерность выходных векторов капсул в этом слое
    :param routings: количество итераций для алгоритма маршрутизации
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Матрица преобразования, от каждой входной капсулы до каждой выходной капсулы
        # есть уникальный вес, как в полносвязном слое.
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)

        # Продублируем размер num_capsule для подготовки к умножению на W
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])

        # Вычислите `входы * W` путем сканирования inputs_tiled на размерности 0.
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
        # Рассматриваем первые два измерения как 'пакетное' измерение, затем
        # matmul(W, x): [..., dim_capsule, input_dim_capsule] x [..., input_dim_capsule, 1] -> [..., dim_capsule, 1].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))

        # Начало алгоритма маршрутизации ---------------------------------------------------------------------#
        # Приоритет для коэффициента сцепления, инициализированный нулями.
        b = tf.zeros(shape=[inputs.shape[0], self.num_capsule, 1, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)

            # c.shape = [batch_size, num_capsule, 1, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # Первые два измерения как `пакетное` измерение,
            # затем matmal: [..., 1, input_num_capsule] x [..., input_num_capsule, dim_capsule] -> [..., 1, dim_capsule].
            # outputs.shape=[None, num_capsule, 1, dim_capsule]
            outputs = squash(tf.matmul(c, inputs_hat))  # [None, 10, 1, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, 1, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # Первые два измерения как `пакетное` измерение, затем
                # matmal:[..., 1, dim_capsule] x [..., input_num_capsule, dim_capsule]^T -> [..., 1, input_num_capsule].
                # b.shape=[batch_size, num_capsule, 1, input_num_capsule]
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)
        # Конец алгоритма марщрутизации -----------------------------------------------------------------------#

        return tf.squeeze(outputs)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Применить Conv2D `n_channels` раз и конкатенировать все капсулы
    :param inputs: 4D тензор, shape=[None, width, height, channels]
    :param dim_capsule: размерность выходного вектора капсулы
    :param n_channels: количество типов капсул
    :return: выходной тензор, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

