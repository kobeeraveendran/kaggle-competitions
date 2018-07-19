from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
np.random.seed(2)

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print(train.head())

train_labels = train['label']
train_images = train.drop(labels = 'label', axis = 1)

del train

print(test.head())
'''

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

#train_images = train_images.reshape((28, 28, 1))
#test_images = test_images.reshape((28, 28, 1))

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size = 0.1, random_state = 2)

model = Sequential([
    Flatten(input_shape = (28, 28)), 
    Dense(512, activation = 'relu'), 
    Dense(10, activation = 'softmax')
    ])
    
model.compile(optimizer = 'rmsprop', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
              
model.fit(x = train_images, y = train_labels, validation_data = (val_images, val_labels), epochs = 5, batch_size = 128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ' + str(test_acc))

kaggle_test = pd.read_csv('dataset/test.csv')
kaggle_test = kaggle_test.values.reshape((-1, 28, 28))

predictions = model.predict(kaggle_test)
print('Prediction for image 5: ' + str(np.argmax(predictions[4])))

submission = open('submission.csv', 'w')

# prepare submission file for Kaggle
submission.write('ImageId,Label')
for i in range(1, len(predictions) + 1):
    submission.write('\n' + str(i) + ',' + str(np.argmax(predictions[i - 1])))

submission.close()