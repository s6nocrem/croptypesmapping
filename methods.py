# import packages
import json
import glob
import tensorflow as tf
from sklearn.utils import class_weight as cw
from tensorflow import keras
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import pandas as pd
import seaborn as sns
import matplotlib
import csv
import matplotlib.pyplot as plt
from imutils import paths

# three functions for getting different orders of mosaics
# random, ordered by class balance, ordered for prediction
def get_random():
    path = "/home/s6nocrem/project/croptypes/csv/mosaics_training/ssh/"
    filepath = glob.glob(path + '*.csv', recursive=True)

    mos_list = []
    overall_list = []

    # random list
    shuffled_list = filepath[:]
    random.shuffle(shuffled_list)

    # print(shuffled_list)

    for i in range(1, len(shuffled_list)+1):
        counter = 0
        while counter < i:
            f = shuffled_list[counter]
            mos_list.append(f)
            counter += 1
        overall_list.append(mos_list[:])
        mos_list = []

    return overall_list, shuffled_list


def get_order():
    path = "/home/s6nocrem/project/croptypes/csv/mosaics_training/ssh/"
    filepath = glob.glob(path + '*.csv', recursive=True)

    mos_list = []
    overall_list = []

    val_list = []
    overall_val_list = []

    with open(path + 'mosaic_order.txt', 'r') as file:
        filepath = json.load(file)

    for i in range(2, len(filepath['order_mosaics'])):
        counter = 1
        while counter < i:
            f = filepath['order_mosaics']['{}'.format(counter)]
            f = f.split('\\')[-1]
            #f = path + f
            mos_list.append(f)
            counter += 1
        overall_list.append(mos_list[:])
        mos_list = []

    return overall_list, "_"


def get_order_pred():
    path = "/home/s6nocrem/project/croptypes/csv/mosaics_training/ssh/"
    filepath = glob.glob(path + '*.csv', recursive=True)

    mos_list = []
    overall_list = []

    val_list = []
    overall_val_list = []

    with open(path + 'mosaic_order.txt', 'r') as file:
        filepath = json.load(file)

    for i in range(2, len(filepath['order_mosaics'])):
        counter = 1
        while counter < i:
            f = filepath['order_mosaics']['{}'.format(counter)]
            f = f.split('\\')[-1]
            #f = path + f
            mos_list.append(f)
            counter += 1
        overall_list.append(mos_list[:])
        mos_list = []
    #p = 37
    p = 40
    counter = 1
    while counter < p:
        f = filepath['order_mosaics']['{}'.format(counter)]
        f = f.split('\\')[-1]
        #f = path + f
        val_list.append(f)
        counter += 1

    for k in range(2, len(filepath['order_mosaics'])):
        counter = 1
        while counter <= k:
            f = filepath['order_mosaics']['{}'.format(counter)]
            f = f.split('\\')[-1]
            #f = path + f
            if f in val_list:
                val_list.remove(f)
            counter += 1
        # print(val_list)
        # print('----')
        overall_val_list.append(val_list[:])

    return overall_list, overall_val_list


# construct a plot that plots and saves the training history
def plot_training(H, N, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

# train a model
def train_model(train_path, extra_path, val_path, dataset_size, dataset_number, net, num_classes, batch_size, epochs, repetition, mode):
# select the GPU on the server and secure memory space
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# set paths according to the mode
    if mode == 'ordered':
        BASE_PATH = "/home/s6nocrem/project/croptypes/model/mosaics_order_rep/{}_mosaic_new".format(
            dataset_number)
    elif mode == 'random':
        BASE_PATH = "/home/s6nocrem/project/croptypes/model/mosaics_rand_rep/{}_mosaic".format(
            dataset_number)
    elif mode == 'other':
        BASE_PATH = "/home/s6nocrem/project/croptypes/model/mosaics_other_rep/{}_mosaic_equal_number".format(
            dataset_number)
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    print(BASE_PATH)
    MODEL_PATH = BASE_PATH + \
        "/model_{}_{}_{}.model".format(repetition, dataset_number, batch_size)
    PLOT_PATH = BASE_PATH + \
        "/model_{}_{}_{}_train.png".format(repetition, dataset_number, batch_size)
    ACC_PATH = BASE_PATH + \
        "/model_{}_{}_{}_acc.png".format(repetition, dataset_number, batch_size)
    REP_PATH = BASE_PATH + \
        "/model_{}_{}_{}_rep.csv".format(repetition, dataset_number, batch_size)

# set further parameters, either for VGG-16 or GoogLeNet with different input shapes
    if net == 'IN3':
        INPUT_SHAPE = (299, 299, 3)
        INPUT_S = (299, 299)
    else:
        INPUT_SHAPE = (224, 224, 3)
        INPUT_S = (224, 224)

    # load csv files and get length for training data
    # print(train_path)
    df_train = []
    for k in train_path:
        df = pd.read_csv(k)
        if mode == 'other':
            df = df[df.label != 'other'] # take out other class
        df_train.append(df)
    df_train = pd.concat(df_train)

    # get equal numbers for all classes based on the smallest class
    """
    def len_crop(crop):
        return df_train[df_train.label ==crop].shape[0]

    minimum_value = 100000

    for i in ('Maize', 'Groundnut', 'Sweet potatoes', 'Tobacco', 'Cassava'):
        if len_crop(i) < minimum_value:
            minimum_value = len_crop(i)

    df_train_subset = []

    for i in ('Maize', 'Groundnut', 'Sweet potatoes', 'Tobacco', 'Cassava'):
        current_crop = df_train[df_train.label == i]
        sample = current_crop.sample(minimum_value)
        df_train_subset.append(sample)

    df_train_subset = pd.concat(df_train_subset)

    df_train = df_train_subset
    """
    
    if extra_path != train_path: # add further training samples from other mosaics
        relevant = ["Cassava", "Groundnut"]

        i = 0
        for p in extra_path:
            if i <= len(train_path):
                i = i + 1
                continue
            elif i > len(train_path):
                i = i+1
                df = pd.read_csv(p)
                df = df[df.label.isin(relevant)]
                df_train.append(df)

    # load csv files and get length for validation data
    df_val = pd.read_csv(val_path)
    if mode == 'other':
        df_val = df_val[df_val.label != 'other']  # take out other class

    train_files = len(df_train.index)
    validate_files = len(df_val.index)
    test_files = len(df_val.index)

    # get class weights for inbalanced classes

    y_train = df_train['label']

    class_weights = cw.compute_class_weight('balanced',
                                            np.unique(y_train),
                                            y_train)
    class_weights = dict(enumerate(class_weights))

    print(class_weights)

    # data preparation

    # initialise data augmentation objects
    train_datagen = ImageDataGenerator(zoom_range=0.3, rotation_range=360,
                                       width_shift_range=30, height_shift_range=30,
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_datagen.rescale = 1./255
    val_datagen.rescale = 1./255
    test_datagen.rescale = 1./255

    # initialize the training generator
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        directory=None,
        x_col="id",
        y_col="label",
        weight_col=None,
        target_size=INPUT_S,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        validate_filenames=False,
        shuffle=True,
    )

    # initialize the validation generator
    val_generator = val_datagen.flow_from_dataframe(
        df_val,
        directory=None,
        x_col="id",
        y_col="label",
        weight_col=None,
        target_size=INPUT_S,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    # initialize the testing generator
    test_generator = test_datagen.flow_from_dataframe(
        df_val,
        directory=None,
        x_col="id",
        y_col="label",
        weight_col=None,
        target_size=INPUT_S,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    # load pre-trained model without the top
    if net == 'IN3':
        model = InceptionV3(include_top=False, input_shape=INPUT_SHAPE, weights="imagenet")
    else:
        model = VGG16(include_top=False, input_shape=INPUT_SHAPE, weights="imagenet")

    # add new classifier layers
    model_new = model.output
    model_new = Flatten(name="flatten")(model_new)
    model_new = Dense(512, activation="relu")(model_new)
    model_new = Dropout(0.5)(model_new)
    model_new = Dense(num_classes, activation="softmax")(model_new)

    # define new model
    model_out = Model(inputs=model.inputs, outputs=model_new)

    # compile model
    print("[INFO] compiling model...")
    opt = SGD(lr=3e-4, momentum=0.9)
    model_out.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

    # train model
    print("[INFO] training network...")
    H = model_out.fit(
        x=train_generator,
        steps_per_epoch=train_files // batch_size,
        validation_data=val_generator,
        validation_steps=validate_files // batch_size,
        epochs=epochs,
        class_weight=class_weights)

    # use trained model to make predictions on the data
    print("[INFO] evaluating after fine-tuning network...")
    predIdxs = model_out.predict(x=test_generator,
                                 steps=(test_files // batch_size) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    # classification report
    cr = classification_report(test_generator.classes, predIdxs,
                               target_names=test_generator.class_indices.keys(),
                               output_dict=True)
    print(cr)
    cr_rep = pd.DataFrame(cr).transpose()
    cr_rep.to_csv(REP_PATH, index=True, header=True)
    # confusion matrix
    cm = confusion_matrix(test_generator.classes, predIdxs)
    if mode == 'other':
        class_list = ['Cassava', 'Groundnut', 'Maize', 'Sweet potatoes', 'Tobacco']
    else:
        class_list = ['Cassava', 'Groundnut', 'Maize', 'Sweet potatoes', 'Tobacco', 'other']
    # heatmap
    cm_df = pd.DataFrame(cm, index=class_list,
                         columns=class_list)
    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.savefig(ACC_PATH)
    # training plot
    plot_training(H, epochs, PLOT_PATH)
    # serialize the model to disk
    print("[INFO] serializing network...")
    model_out.save(MODEL_PATH, save_format="h5")
    print(MODEL_PATH)
