import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D,Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

np.random.seed(2)
sns.set(style = 'white')

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

y_train = train['label']
x_train = train.drop(labels = ["label"], axis = 1)

del train

graph = sns.countplot(y_train)

print(y_train.value_counts())
plt.show()