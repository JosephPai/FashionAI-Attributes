import gc
from tqdm import tqdm
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.applications.densenet import preprocess_input
from .dataset import *


def getX(n, x_path):
    X = np.zeros((n, image_size, image_size, 3), dtype=np.uint8)
    for i in tqdm(range(n)):
        X[i] = padding(x_path[i])
    return X


def predict():
    x_path, y = create_dataset(TEST_PATH % task_name)
    num_classes = len(y[0])
    n = len(x_path)
    X = getX(n, x_path)

    cnn_model = DenseNet121(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')
    inputs = Input((image_size, image_size, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs, x)

    model.load_weights(model_name)
    test_np = model.predict(X, batch_size=256)
    np.savetxt(SAVE_LABEL_PATH, test_np)

    del model
    del X
    gc.collect()


if __name__=="__main__":
    predict()