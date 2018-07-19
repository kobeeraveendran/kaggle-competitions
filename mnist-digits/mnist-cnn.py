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
from keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout
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
    MaxPool2D(strides = (2,2)), 
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

NUM_EPOCHS = 2
BATCH_SIZE = 86

# data augmentation
datagen = ImageDataGenerator(featurewise_center = False, 
                             samplewise_center = False, 
                             featurewise_std_normalization = False, 
                             samplewise_std_normalization = False, 
                             zca_whitening = False, 
                             rotation_range = 10, 
                             zoom_range = 0.1, 
                             width_shift_range = 0.1, 
                             height_shift_range = 0.1, 
                             horizontal_flip = False, 
                             vertical_flip = False)

# fit model
generator = datagen.flow(x_train, y_train, BATCH_SIZE)
history = model.fit_generator(generator, 
                    epochs = NUM_EPOCHS, 
                    validation_data = (x_val, y_val), 
                    steps_per_epoch = x_train.shape[0] // BATCH_SIZE, 
                    callbacks = [learning_rate_reduction], 
                    verbose = 2)

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color = 'b', label = 'Training loss')
ax[0].plot(history.history['val_loss'], color = 'r', label = 'Validation loss', axes = ax[0])
legend = ax[0].legend(loc = 'best', shadow = True)
ax[1].plot(history.history['acc'], color = 'b', label = 'Training accuracy')
ax[1].plot(history.history['val_acc'], color = 'r', label = 'Validation accuracy')
legend = ax[1].legend(loc = 'best', shadow = True)


# plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation = 45)
    plt.yticks(ticks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    threshold = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment = 'center', color = 'white' if cm[i,j] > threshold else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true =np.argmax(y_val, axis = 1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(10))

