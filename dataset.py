import csv
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from .config import *

label_dict_one_hot = {"n": 0, "y": 1, "m": 0}
label_dict_soft1 = {"n": 0, "y": 0.75, "m": 0.25}
label_dict_soft2 = {"n": 0, "y": 0.8, "m": 0.1}
label_dict_soft3 = {"n": 0, "y": 0.7, "m": 0.1}
attributes = ["skirt_length_labels", "coat_length_labels", "pant_length_labels", "sleeve_length_labels",
              "collar_design_labels", "lapel_design_labels", "neck_design_labels", "neckline_design_labels"]


# generate single_task's label before train
def get_single_task_soft_label(attr):
    single_task_label_path = TRAIN_PATH % str(attr).replace("_labels", "")
    single_task_label = csv.writer(open(single_task_label_path, "w", newline=""))
    counter = 1
    with open(TRAIN_PATH % "alllabel") as f:
        reader = csv.reader(f)
        for line in reader:
            new_line_one_hot = list()
            new_line_soft1, new_line_soft2, new_line_soft3 = list(), list(), list()
            new_line_one_hot.append(line[0])
            new_line_soft1.append(line[0])
            new_line_soft2.append(line[0])
            new_line_soft3.append(line[0])
            for c in list(line[-1]):
                new_line_one_hot.append(label_dict_one_hot[c])
                new_line_soft1.append(label_dict_soft1[c])
                new_line_soft2.append(label_dict_soft2[c])
                new_line_soft3.append(label_dict_soft3[c])
            if line[1] == attr:
                tmp = line[-1].count("m")
                if tmp == 0:
                    single_task_label.writerow(new_line_one_hot)
                elif tmp == 1:
                    single_task_label.writerow(new_line_soft1)
                elif tmp == 2:
                    single_task_label.writerow(new_line_soft2)
                elif tmp == 3:
                    single_task_label.writerow(new_line_soft3)
            counter = counter + 1
    print(attr, "Count: ", counter)


# below: dataset and image utils for training
def create_dataset(label_path):
    label_raw = []
    all_images = []
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            all_images.append(line[0])
            label_raw.append(line[1:])
    return np.array(all_images), np.array(label_raw, dtype=np.float16)


def padding(filename):
    img = cv2.imread(filename)[:, :, ::-1]      # BGR TO RGB
    old_size = img.shape[:2]
    ratio = float(image_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = image_size - new_size[1]
    delta_h = image_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


class Generator():
    def __init__(self, X, y, batch_size=32, aug=False):
        def generator():
            idg = ImageDataGenerator(channel_shift_range=60,
                                     rotation_range=30,
                                     horizontal_flip=True,
                                     shear_range=0.30,
                                     zoom_range=0.23)
            while True:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].copy()
                    y_batch = y[i:i+batch_size].copy()
                    if aug:
                        for j in range(len(X_batch)):
                            X_batch[j] = idg.random_transform(X_batch[j])
                    yield X_batch, y_batch
        self.generator = generator()
        self.steps = len(X) // batch_size + 1


if __name__=='__main__':
    for attr in attributes:
        get_single_task_soft_label(attr)
