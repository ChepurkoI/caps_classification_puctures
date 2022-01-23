# CapsNet-Keras
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/XifengGuo/CapsNet-Keras/blob/master/LICENSE)

Реализация Keras/TensorFlow2.2 для CapsNet в статье:   
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   
Текущая `средняя ошибка теста = 0.34%` и `лучшая ошибка теста = 0.30%`.   
 
**Отличия от статьи:**.   
- Мы используем затухание скорости обучения с `decay factor = 0.9` и `step = 1 epoch`,    
в то время как в статье не указаны подробные параметры (или они их не использовали?).
- Мы сообщаем только об ошибках теста после `50 эпох` обучения.   
В статье, я полагаю, они тренировались в течение `1250 эпох` в соответствии с рисунком A.1?
Звучит безумно, возможно, я неправильно понял.
- Мы используем MSE (средняя квадратичная ошибка) в качестве потери при реконструкции и 
коэффициент для этой потери - `lam_recon=0.0005*784=0.392`.   
Это должно быть **эквивалентно** использованию SSE (sum squared error) и `lam_recon=0.0005`, как в статье.


**ТОДО**
- Провести эксперименты на других наборах данных. 
- Исследуйте интересные характеристики CapsuleNet.

**Контакты**
- Ваш вклад в репозиторий всегда приветствуется. 
Откройте проблему или свяжитесь со мной по E-mail `guoxifeng1990@163.com` или WeChat `wenlong-guo`.


## Использование

**Шаг 1.
Установите бэкенд [TensorFlow>=2.0](https://github.com/tensorflow/tensorflow).**.
```
pip install tensorflow==2.2.0
```

**Шаг 2. Клонируйте этот репозиторий на локальный.**.
```
git clone https://github.com/XifengGuo/CapsNet-Keras.git capsnet-keras
cd capsnet-keras
```

**Этап 3. Обучение CapsNet на MNIST**.  

Обучение с настройками по умолчанию:
```
python capsulenet.py
```

Более подробный запуск для справки:
```
python capsulenet.py -h
```

**Шаг 4. Тестирование предварительно обученной модели CapsNet**.

Предположим, что вы обучили модель с помощью вышеприведенной команды, тогда обученная модель будет сохранена в файле `result/trained_model.h5`.
сохранена в `result/trained_model.h5`. Теперь просто запустите следующую команду, чтобы получить результаты тестирования.
```
$ python capsulenet.py -t -w result/trained_model.h5
```
Она выведет точность тестирования и покажет восстановленные изображения.
Данные для тестирования такие же, как и для валидации. Это облегчит тестирование на новых данных, 
просто измените код по своему усмотрению.

Вы также можете просто *скачать модель, которую я тренировал* с сайта 
https://pan.baidu.com/s/1sldqQo1
или
https://drive.google.com/open?id=1A7pRxH7iWzYZekzr-O0nrwqdUUpUpkik


**Шаг 5. Обучение на нескольких процессорах**   

Для этого требуется `Keras>=2.0.9`. После обновления Keras:   
```
python capsulenet-multi-gpu.py --gpus 2
```
Он автоматически обучается на нескольких процессорах в течение 50 эпох, а затем выводит производительность на тестовом наборе данных.
Но во время обучения не сообщается о точности валидации.

## Результаты

#### Ошибки теста   

Тест классификации CapsNet **ошибка** на MNIST. Среднее значение и стандартное отклонение результатов
представлены по 3 испытаниям. Результаты можно воспроизвести, запустив следующие команды.   
 ```
 python capsulenet.py --routings 1 --lam_recon 0.0 #CapsNet-v1   
 python capsulenet.py --routings 1 --lam_recon 0.392 #CapsNet-v2
 python capsulenet.py --routings 3 --lam_recon 0.0 #CapsNet-v3 
 python capsulenet.py --routings 3 --lam_recon 0.392 #CapsNet-v4
```
   Метод | Маршрутизация | Реконструкция | MNIST (%) | *Paper*    
   :---------|:------:|:---:|:----:|:----:
   Базовый уровень | -- | -- | -- | -- | | *0.39*. 
   CapsNet-v1 | 1 | нет | 0.39 (0.024) | *0.34 (0.032)* 
   CapsNet-v2 | 1 | да | 0,36 (0,009)| *0,29 (0,011)*
   CapsNet-v3 | 3 | нет | 0,40 (0,016) | *0,35 (0,036)*
   CapsNet-v4 | 3 | да| 0.34 (0.016) | *0.25 (0.005)*
   
Потери и точность:   
![](result/log.png)


#### Скорость обучения 

Около `100 с / эпоха` на одном графическом процессоре GTX 1070.   
Около `80 с / эпоха` на одном графическом процессоре GTX 1080Ti.   
Около `55s / epoch` на двух GTX 1080Ti GPU при использовании `capsulenet-multi-gpu.py`.      

#### Результат реконструкции  

Результат CapsNet-v4 при запуске   
```
python capsulenet.py -t -w result/trained_model.h5
```
Цифры в верхних 5 строках - реальные изображения из MNIST, а 
цифры внизу - соответствующие реконструированные изображения.

![](result/real_and_recon.png)

#### Манипулирование латентным кодом

```
python capsulenet.py -t --digit 5 -w result/trained_model.h5 
```
Для каждой цифры *i*-я строка соответствует *i*-му измерению капсулы, а столбцы слева направо соответствуют сложению `[-0,5]. 
справа соответствуют добавлению `[-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]` к 
значение одного измерения капсулы. 

Как мы видим, каждое измерение уловило некоторые характеристики цифры. Одно и то же измерение 
разных капсул цифр может представлять разные характеристики. Это связано с тем, что разные 
цифры восстанавливаются из разных векторов признаков (капсул цифр). Эти векторы взаимно 
независимы во время реконструкции.

Переведено с помощью www.DeepL.com/Translator (бесплатная версия)    
![](result/manipulate-0.png)
![](result/manipulate-1.png)
![](result/manipulate-2.png)
![](result/manipulate-3.png)
![](result/manipulate-4.png)
![](result/manipulate-5.png)
![](result/manipulate-6.png)
![](result/manipulate-7.png)
![](result/manipulate-8.png)
![](result/manipulate-9.png)


## Other Implementations

- PyTorch:
  - [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  I referred to some functions in this repository.
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)

- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)
