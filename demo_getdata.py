import os
import shutil
import sys

import keras

TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, VAL_IMAGE_PATH, VAL_LABEL_PATH, TEST_IMAGE_PATH, TEST_LABEL_PATH = sys.argv[1:7]

def train_source():
    yield '/external/path/to/image', 'label1'
    yield '/external/path/to/image', 'label2'

def val_source():
    yield '/external/path/to/image', 'label1'
    yield '/external/path/to/image', 'label2'

def populate_dir(source, destsuperdir):
    n = 0
    for imagepath, label in train_source():
        destdir = os.path.join(destsuperdir, label)
        if not os.path.isdir(destdir): os.mkdir(destdir)
        shutil.copyfile(imagepath, os.path.join(destdir, os.path.basename(imagepath)))
        n += 1
    return n

def predict_features(model, imagedir, N, batchsize=1)
    imdata = keras.preprocessing.image.ImageDataGenerator()
    flow = imdata.flow_from_directory(imagedir,
        target_size=(imwidth, imheight),
        batch_size=batchsize,
        class_mode=None,
        shuffle=False)
    features = model.predict_generator(flow, N, verbose=1)
    return features

# Organize data for keras
N_train = populate_dir(train_source, TRAIN_IMAGE_PATH)
N_val = populate_dir(val_source, VAL_IMAGE_PATH)

# Compute features
imwidth,imheight = 640,640
feature_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(imwidth,imheight,3))
train_features = predict_features(feature_model, TRAIN_IMAGE_PATH, N_train) #for batchsize=1
val_features = predict_features(feature_model, VAL_IMAGE_PATH, N_val) #for batchsize=1