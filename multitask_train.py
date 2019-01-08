import os
import cv2
import random
from tqdm import tqdm
from glob import glob
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.callbacks import *
from keras.regularizers import l2
from keras.preprocessing.image import *
from keras.utils import multi_gpu_model
import keras_applications
from keras import backend, layers, models, utils
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
model_name = 'multi_length'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 读取数据
print("Reading data")
df = pd.read_csv(r"/home/wubin/Tianchi/Annotations/soft/train/multilen_train.csv", header=None)
df.columns = ['filename', 'label_name', 'label']
df = df.sample(frac=1).reset_index(drop=True)       # sample起到了shuffle的作用
df.label_name = df.label_name.str.replace('_labels', '')
c = Counter(df.label_name)
label_count = dict([(x, len(df[df.label_name == x].label.values[0])) for x in c.keys()])
label_names = list(label_count.keys())

# 生成label
label_dict_one_hot = {"n": 0, "y": 1, "m": 0}
label_dict_soft1 = {"n": 0, "y": 0.75, "m": 0.25}
label_dict_soft2 = {"n": 0, "y": 0.8, "m": 0.1}
label_dict_soft3 = {"n": 0, "y": 0.7, "m": 0.1}


def get_soft_label(label):
    label = str(label)
    count = label.count('m')
    label_one_hot, label_soft1, label_soft2, label_soft3 = list(), list(), list(), list()
    for c in list(label):
        label_one_hot.append(label_dict_one_hot[c])
        label_soft1.append(label_dict_soft1[c])
        label_soft2.append(label_dict_soft2[c])
        label_soft3.append(label_dict_soft3[c])
    if count==0:
        return label_one_hot
    elif count==1:
        return label_soft1
    elif count==2:
        return label_soft2
    elif count==3:
        return label_soft3


fnames = df['filename'].values
width = 448
n = len(df)
y = [np.zeros((n, label_count[x])) for x in label_count.keys()]
print(len(y), y[0].shape, y[-1].shape)
for i in range(n):
    label_name = df.label_name[i]
    label = df.label[i]
    soft_label = get_soft_label(label)
    y[label_names.index(label_name)][i] = soft_label


def f(index):
    return index, cv2.resize(cv2.imread(fnames[index]), (width, width))


# 读取图片，生成data
print("Reading images")
X = np.zeros((n, width, width, 3), dtype=np.uint8)
with multiprocessing.Pool(12) as pool:
    with tqdm(pool.imap_unordered(f, range(n)), total=n) as pbar:
        for i, img in pbar:
            X[i] = img[:, :, ::-1]

n_train = int(n*0.85)    # 前面已经打乱过了
X_train = X[:n_train]
X_valid = X[n_train:]
y_train = [x[:n_train] for x in y]
y_valid = [x[n_train:] for x in y]


class Generator():
    def __init__(self, X, y, batch_size=32, aug=False):
        def generator():
            idg = ImageDataGenerator(horizontal_flip=True,
                                     rotation_range=20,
                                     zoom_range=0.2,
                                     channel_shift_range=50)
            while True:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].copy()
                    y_batch = [x[i:i+batch_size] for x in y]
                    if aug:
                        for j in range(len(X_batch)):
                            X_batch[j] = idg.random_transform(X_batch[j])
                    yield X_batch, y_batch
        self.generator = generator()
        self.steps = len(X) // batch_size+1


gen_train = Generator(X_train, y_train, batch_size=32, aug=True)


def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)


base_model = DenseNet121(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')
input_tensor = Input((width, width, 3))
x = input_tensor
x = Lambda(densenet.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.5)(x)
x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]
model = Model(input_tensor, x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model2 = multi_gpu_model(model, n_gpus)
model2.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=[acc])
model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps,
                     epochs=4, validation_data=(X_valid, y_valid), callbacks=[EarlyStopping(patience=2)])

model.save('model_%s.h5' % model_name)

y_pred = model2.predict(X_valid, batch_size=32, verbose=1)
a = np.array([x.any(axis=-1) for x in y_valid]).T.astype('uint8')
b = [np.where((a == np.eye(8)[x]).all(axis=-1))[0] for x in range(8)]
for c in range(4):
    y_pred2 = y_pred[c][b[c]].argmax(axis=-1)
    y_true2 = y_valid[c][b[c]].argmax(axis=-1)
    print(label_names[c], (y_pred2 == y_true2).mean())
