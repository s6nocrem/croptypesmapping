# import packages
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import os
from methods import get_order, train_model

# get ordered list of mosaics for testing
_, test_all = get_order_pred()
# put number of mosaics - 1
test_path = test_all[24]

# load the trained model from disk
# select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# set paths
BASE_PATH = "/home/s6nocrem/project/croptypes/model/mosaics_other_rep/21_mosaic_extra"
if not os.path.exists(BASE_PATH + "/pred"):
    os.makedirs(BASE_PATH + "/pred")
for i in range(5):
    # load model
    model = load_model(BASE_PATH + "/model_{}_21_32.model".format(i))
    # set paths
    ACC_PATH = BASE_PATH + "/pred/model_prediction_{}.png".format(i)
    REP_PATH = BASE_PATH + "/pred/model_prediction_{}.csv".format(i)
    CSV_PATH = BASE_PATH + "/pred/model_acc_{}.csv".format(i)
    # set input shapes
    INPUT_SHAPE = (224, 224, 3)
    INPUT_S = (224, 224)

    # prepare test data
    test_datagen = ImageDataGenerator()

    print(test_path)
    df_test = []
    for k in test_path:
        df = pd.read_csv(k)
        df_test.append(df)

    df_test = pd.concat(df_test)

    print(df_test)

    # initialize the testing generator
    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        directory=None,
        x_col="id",
        y_col="label",
        weight_col=None,
        target_size=INPUT_S,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
    )

    # encode = {0: "Cassava", 1: 'Groundnut', 2: 'Maize',
    #          3: 'Sweet potatoes', 4: 'Tobacco', 5: 'other'}
    encode = {0: "Cassava", 1: 'Groundnut', 2: 'Maize',
              3: 'Sweet potatoes', 4: 'Tobacco'}

    pred = model.predict(test_generator,
                         steps=len(test_generator), verbose=1)

    # Get classes by np.round
    cl = np.argmax(pred, axis=1)
    count = 0
    # Get filenames (set shuffle=false in generator is important)
    filenames = test_generator.filenames

    results = []

    for f, c, p in zip(filenames, cl, pred):
    # select for higher confidence:
        true_label = f.split('/')[-2]
        results.append([f, np.max(p), encode[c], true_label])
        if encode[c] != true_label:
            count += 1
    print(count)
    # write to file
    results_df = pd.DataFrame(
        results, columns=['file', 'probability', 'predicted class', 'true class'])
    print(results_df)
    results_df.to_csv(REP_PATH, index=True, header=True)
    # get classification report
    cr = classification_report(test_generator.classes, cl,
                               target_names=test_generator.class_indices.keys(),
                               output_dict=True)
    print(cr)
    cr_rep = pd.DataFrame(cr).transpose()
    cr_rep.to_csv(CSV_PATH, index=True, header=True)
    # get confusion matrix 
    cm = confusion_matrix(test_generator.classes, cl)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #class_list = ['Cassava', 'Groundnut', 'Maize', 'Sweet potatoes', 'Tobacco', 'other']
    class_list = ['Cassava', 'Groundnut', 'Maize', 'Sweet potatoes', 'Tobacco']
    cm_df = pd.DataFrame(cm, index=class_list,
                         columns=class_list)
    plt.figure(figsize=(5.5, 4))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    # plt.show()
    plt.savefig(ACC_PATH)
