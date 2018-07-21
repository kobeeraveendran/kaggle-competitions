import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from sklearn.linear_model import LogisticRegression

start = dt.datetime.now()

INPUT_SIZE = 224
NUM_CLASSES = 40
SEED = 1987

labels = pd.read_csv('dataset/kaggle/labels.csv')

selected_breed_list = list(labels.groupby('breed').count().sort_values(by = 'id', ascending = False).head(NUM_CLASSES).index)

labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

np.random.seed(seed = SEED)
rand = np.random.random(len(labels))

# split into training/validation sets
train_index = rand < 0.8
valid_index = rand >= 0.8
y_train = labels_pivot[selected_breed_list].values

ytr = y_train[train_index]
yv = y_train[valid_index]

