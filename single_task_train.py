import gc
from tqdm import tqdm
import multiprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.utils import multi_gpu_model
from keras.applications.densenet import preprocess_input
from .dataset import *


def getX(n, x_path):
    X = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
    # for i in tqdm(range(n)):
    #     X[i] = padding(x_path[i])
    with multiprocessing.Pool(12) as pool:
        with tqdm(pool.imap_unordered(padding, x_path), total=n) as pbar:
            for i, img in pbar:
                X[i] = img
    return X


def train():
    x_path, y = create_dataset(TRAIN_PATH % task_name)
    print("Train shape: ", x_path.shape)
    print("Label shape: ", y.shape)

    num_classes = len(y[0])
    n = len(x_path)
    X = getX(n, x_path)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, shuffle=True)
    print("Image train shape:", X_train.shape, "Label train shape", y_train.shape)
    print("Image val shape:", X_valid.shape, "Label val shape", y_valid.shape)
    gen_train = Generator(X_train, y_train, batch_size=32, aug=True)

    cnn_model = DenseNet121(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
    inputs = Input((image_size, image_size, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs, x)

    opt = Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model = multi_gpu_model(model, gpus=len(gpu_list))
    parallel_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=10,
                                 callbacks=[EarlyStopping(patience=2)], shuffle=True,
                                 validation_data=(X_valid, y_valid))

    parallel_model.compile(optimizer=Adam(3e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=2,
                                 shuffle=True, validation_data=(X_valid, y_valid))

    parallel_model.compile(optimizer=SGD(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1,
                                 shuffle=True, validation_data=(X_valid, y_valid))

    parallel_model.compile(optimizer=SGD(3e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    parallel_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1,
                                 shuffle=True, validation_data=(X_valid, y_valid))
    model.save("model_%s.h5" % model_name)

    test_np = model.predict(X_valid, batch_size=256)
    y_pred, y_true = list(), list()
    for i in range(len(test_np)):
        y_pred.append(np.argmax(test_np[i]))
        y_true.append(np.argmax(y_valid[i]))
    print("Accuracy: ", metrics.accuracy_score(y_true, y_pred))
    print("Precision: ", metrics.precision_score(y_true, y_pred, average='macro'))
    print("Recall: ", metrics.recall_score(y_true, y_pred, average='macro'))
    print("F1 Score: ", metrics.f1_score(y_true, y_pred, average='weighted'))
    print(metrics.classification_report(y_true, y_pred, target_names=attributes))

    del model
    del parallel_model
    del X
    gc.collect()


if __name__=="__main__":
    train()