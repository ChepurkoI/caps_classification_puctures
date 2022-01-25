import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical, image_dataset_from_directory

import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):
    """
     Капсульная сеть.
    :param input_shape: форма данных, 3d, [width, height, channels]
    :param n_class: количество классов
    :param routings: количество итераций маршрутизации
    :param batch_size: размер партии
    :return: Две модели Keras, первая из которых используется для обучения, а вторая - для оценки.
            `eval_model` также может быть использована для обучения.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # Layer 1: Обычный слой Conv2D
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Слой Conv2D с активацией `squash`, затем переформирование в [None, num_capsule, dim_capsule].
    primarycaps = PrimaryCap(conv1, dim_capsule=16, n_channels=16, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Капсульный слой. Здесь работает алгоритм маршрутизации.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: Это вспомогательный слой для замены каждой капсулы ее длиной.
    # Просто чтобы соответствовать истинной форме метки.
    out_caps = Length(name='capsnet')(digitcaps)

    # Сеть декодеров.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # Истинная метка используется для маски выхода капсульного слоя. Для обучения
    masked = Mask()(digitcaps)  # Маска с использованием капсулы с максимальной длиной. Для предсказания

    # Общая модель декодера при обучении и прогнозировании
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=32 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Модели для обучения и оценки (прогнозирование)
    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # манипулирование моделью
    noise = layers.Input(shape=(n_class, 32))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.

    Маржинальная потеря для уравнения (4). Когда y_true[i, :] содержит не только один `1`,
    эта потеря тоже должна работать.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: скалярное значение потерь.
    """

    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model,  # type: models.Model
          data, args):
    """
    Обучение сети CapsuleNet
    :param model: модель CapsuleNet
    :param data: кортеж, содержащий данные для обучения и тестирования, например `((x_train, y_train), (x_test, y_test))`.
    :param args: аргументы
    :return: Обученная модель
    """

    # распаковка данных
    (x_train, y_train), (x_test, y_test) = data

    # обратные вызовы
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (args.lr_decay ** epoch))

    # компиляция модели
    model.compile(optimizer=optimizers.Adam(learning_rate=args.learning_rate),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Обучение без увеличения данных:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Начало: Обучение с увеличением данных ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # сдвиг на 2 пикселя
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch), (y_batch, x_batch)

    # Обучение с увеличением данных. Если shift_fraction=0., аугментация отсутствует.
    model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(y_train.shape[0] / args.batch_size),
              epochs=args.epochs,
              validation_data=((x_test, y_test), (y_test, x_test)), batch_size=args.batch_size,
              callbacks=[log, checkpoint, lr_decay])
    # Конец: Обучение с увеличением данных -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def load_data():
    size_train_dataset = 9000
    size_test_dataset = 1500
    ##------Загружаем наш dataset
    img_height = 28
    img_width = 28
    train_dataset = tf.keras.utils.image_dataset_from_directory("NewNameRGB", validation_split=0,
                                                                seed=13,
                                                                image_size=(img_height, img_width),
                                                                batch_size=size_train_dataset, color_mode="rgb",
                                                                shuffle=True)

    validation_dataset = tf.keras.utils.image_dataset_from_directory("NewNameRGB", validation_split=0.2,
                                                                     subset='validation', seed=13,
                                                                     image_size=(img_height, img_width),
                                                                     batch_size=size_train_dataset, color_mode="rgb",
                                                                     shuffle=True)

    test_dataset = tf.keras.utils.image_dataset_from_directory("testRGB", validation_split=0,
                                                               seed=13,
                                                               image_size=(img_height, img_width),
                                                               batch_size=size_test_dataset, color_mode="rgb",
                                                               shuffle=True)

    class_names = train_dataset.class_names
    class_names_test = test_dataset.class_names

    ##--------------------
    class_names = train_dataset.class_names


    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    ##-------------------

#    for image_batch, labels_batch in train_dataset:
#        print(image_batch.shape)
#        print(labels_batch.shape)
#        break

#    for image_batch_test, labels_batch_test in test_dataset:
#        print(image_batch_test.shape)
#        print(labels_batch_test.shape)
#        break

    # -------------------

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    normalized_ds_test = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    image_batch, labels_batch = next(iter(normalized_ds))
    image_batch_test, labels_batch_test = next(iter(normalized_ds_test))

    # --------------------
    image_batch = np.array(image_batch)
    image_batch_test = np.array(image_batch_test)
    image_batch = image_batch.reshape(-1, 28, 28, 3).astype('float32')
    image_batch_test = image_batch_test.reshape(-1, 28, 28, 3).astype('float32')

    labels_batch = np.array(labels_batch)
    labels_batch_test = np.array(labels_batch_test)

    labels_batch = to_categorical(labels_batch, len(class_names), 'float32')
    labels_batch_test = to_categorical(labels_batch_test, len(class_names), 'float32')



    #plt.figure(figsize=(10, 10))
    #for i in range(25):
    #    plt.subplot(5, 5, i + 1)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.imshow(image_batch_test[i], cmap=plt.cm.binary)
    #plt.show()



    ##--------------------возвращаем данные каждой картинки и ее метку
    return (image_batch, labels_batch), (image_batch_test, labels_batch_test)


if __name__ == "__main__":
    import os
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks

    # установка гиперпараметров
    parser = argparse.ArgumentParser(description="Capsule Network.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by learning_rate at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # загружаем данные
    (x_train, y_train), (x_test, y_test) = load_data()

    # определить модель
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings,
                                                  batch_size=args.batch_size)
    model.summary()
    """
    # обучение или тестирование
    if args.weights is not None:  # инициировать веса модели с предоставленными весами
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # пока заданы веса, будет проводиться тестирование
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test), args=args)
"""

