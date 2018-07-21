import cv2
import matplotlib.pyplot as plt
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm
import numpy as np

# dog isolation and detection

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

ResNet50_model = ResNet50(weights = 'imagenet')

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size = (224, 224))
    x = image.img_to_array(img)
    
    return np.expand_dims(x, axis = 0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))

    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)

    return ((prediction <= 268) & (prediction >= 151))

# breed prediction CNN

# load datasets
def load_dataset(path):
	data = load_files(path)
	dog_files = np.array(data['filenames'])
	dog_targets = np_utils.to_categorical(np.array(data['target']), 133)

	return dog_files, dog_targets

train_files, train_targets = load_dataset('dataset/dogImages/train')
valid_files, valid_targets = load_dataset('dataset/dogImages/valid')
test_files, test_targets = load_dataset('dataset/dogImages/test')

# extract bottleneck features
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_DogResnet50 = bottleneck_features['train']
valid_DogResnet50 = bottleneck_features['valid']
test_DogResnet50 = bottleneck_features['test']

# define model, adding a few layers to ResNet
Resnet50_model = Sequential()
Resnet50_model.add(keras.layers.GlobalAveragePooling2D(input_shape = train_DogResnet50.shape[1:]))
Resnet50_model.add(keras.layers.Dense(133, activation = 'softmax'))

Resnet50_model.summary()

Resnet50_model.compile(optimizer = 'rmsprop', 
					   loss = 'categorical_crossentropy', 
					   metrics = ['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(
		'saved_models/weights.best.ResNet50.hdf5', 
		verbose = 1, 
		save_best_only = True
)

Resnet50_model.fit(train_DogResnet50, train_targets, 
	validation_data = (valid_DogResnet50, valid_targets), 
	epochs = 20, batch_size = 20, callbacks = [checkpoint])

Resnet50_model.load_weights('saved_models/weights.best.ResNet50.hdf5')
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis = 0))) for feature in test_DogResnet50]

test_acc = np.sum(np.array(Resnet50_predictions) == np.argmax(test_targets, axis = 1)) / len(Resnet50_predictions) * 100
print('Test Accuracy: ' + str(test_acc))