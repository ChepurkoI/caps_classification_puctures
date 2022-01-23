import cv2
import os
from PIL import Image

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical

format = ".png"  # Формат изображения


def resize_image(input_image_path, output_image_path):
    """
    Изменяет размер изображения на 28*28
    :input_image_path - откуда берётся фото
    :output_image_path - куда сохраняется фото
    """
    size = (64, 64)  # Размер после преобразования

    original_image = Image.open(input_image_path)  # Открыли фото

    resized_image = original_image.resize(size)  # Изменили размер
    # resized_image.show() # Вывели на экран
    resized_image.save(output_image_path)  # Сохранили


def modification_photo(dir_train_old,dir_train_new, dir_test_old,dir_test_new):
    # 'apple','apricot','avocado','banana','carrot','grape','kiwi','onion','orange','pear','pepper','potato','strawberry','tomato','watermelon'
    class_name = ['apple','apricot','avocado','banana','carrot','grape','kiwi','onion','orange','pear','pepper','potato','strawberry','tomato','watermelon']  # Имена классов, они же имена папок, они же подписи фоток
    for j in range(len(class_name)):  # За одну итерацию переименовываются фотки одного класса
        # os.getcwd()
        collection = dir_test_old + "/" + class_name[j]  # Папка с фото (НЕизмененные)
        #/OOP/Fresh_dataset_original
        #OOP/CapsNet-Keras-tf2.2/Test
        new_collection = dir_test_new + "/" + class_name[j]  # Папка с фото (измененные)
        #OOP/CapsNet-Keras-tf2.2/NewNameRGB
        #OOP/CapsNet-Keras-tf2.2/testRGB

        for i, filename in enumerate(os.listdir(collection)):  # Пробегаем по всем фоткам данного класса class_name[j]
            old_name = collection + "/" + filename  # Исходное расположение файла + старое имя
            new_name = new_collection + "/" + class_name[j] + "_" + str(i) + format  # Новое расположение файла + новое имя
            image = cv2.imread(old_name)  # Считали фото и сделали его серым
            cv2.imwrite(new_name, image)  # Записали серое фото в новую папку под новым именем
            # cv2.imshow("Подпись окна", image) # Если вдруг захочется вывести фото, чтобы посмотреть
            # cv2.waitKey(0) # Чтобы выведенное фото закрылось только после нажатия какой-то клавиши
            resize_image(new_name, new_name)  # Изменение размера фотографии

            # os.rename(new_name, new_name) # Переименовали фото: ИмяКласса_НомерФото.Формат

    print("Тестовый набор собран")

    for j in range(len(class_name)):  # За одну итерацию переименовываются фотки одного класса
        # os.getcwd()
        collection = dir_train_old + "/" + class_name[j]  # Папка с фото (НЕизмененные)
        #/OOP/Fresh_dataset_original
        #OOP/CapsNet-Keras-tf2.2/Test
        new_collection = dir_train_new + "/" + class_name[j]  # Папка с фото (измененные)
        #OOP/CapsNet-Keras-tf2.2/NewNameRGB
        #OOP/CapsNet-Keras-tf2.2/testRGB

        for i, filename in enumerate(os.listdir(collection)):  # Пробегаем по всем фоткам данного класса class_name[j]
            old_name = collection + "/" + filename  # Исходное расположение файла + старое имя
            new_name = new_collection + "/" + class_name[j] + "_" + str(i) + format  # Новое расположение файла + новое имя
            image = cv2.imread(old_name)  # Считали фото и сделали его серым
            cv2.imwrite(new_name, image)  # Записали серое фото в новую папку под новым именем
            # cv2.imshow("Подпись окна", image) # Если вдруг захочется вывести фото, чтобы посмотреть
            # cv2.waitKey(0) # Чтобы выведенное фото закрылось только после нажатия какой-то клавиши
            resize_image(new_name, new_name)  # Изменение размера фотографии

            # os.rename(new_name, new_name) # Переименовали фото: ИмяКласса_НомерФото.Формат

dir_train_old="/home/igor/Домашки/OOP/Fresh_dataset_original"
dir_train_new="/home/igor/Домашки/OOP/CapsNet-Keras-tf2.2/NewNameRGB"
dir_test_old="/home/igor/Домашки/OOP/CapsNet-Keras-tf2.2/Test"
dir_test_new="/home/igor/Домашки/OOP/CapsNet-Keras-tf2.2/testRGB"
modification_photo(dir_train_old,dir_train_new, dir_test_old,dir_test_new)

