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

#graph = sns.countplot(y_train)

print(y_train.value_counts())

# check data for null/missing vals
#print(x_train.isnull().any().describe())
#print(test.isnull().any().describe())

# restrict pixel values (grayscale normalization)
x_train = x_train / 255.0
test = test / 255.0

# reshape (flatten) for cnn
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# categorize labels
y_train = to_categorical(y_train, num_classes = 10)

# train validation split (90% train 10% val)
random_seed = 2
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = random_seed)

graph = plt.imshow(x_train[0][:, :, 0])
plt.show()

model = Sequential([
    Conv2D(32, (5,5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)), 
    Conv2D(32, (5,5), padding = 'Same', activation = 'relu'), 
    MaxPool2D(), 
    Dropout(rate = 0.25), 
    Conv2D(64, (3,3), padding = 'Same', activation = 'relu'), 
    Conv2D(64, (3,3), padding = 'Same', activation = 'relu'), 
    MaxPool2D(stides = (2,2)), 
    Dropout(rate = 0.25), 
    Flatten(), 
    Dense(256, activation = 'relu'), 
    Dropout(rate = 0.5), 
    Dense(10, activation = 'softmax')
])

model.compile(optimizer = RMSprop(epsilon = 1e-08), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 
                                            patience = 3, 
                                            factor = 0.5, 
                                            min_lr = 0.00001, 
                                            verbose = 1)

NUM_EPOCHS = 1
BATCH_SIZE = 86
