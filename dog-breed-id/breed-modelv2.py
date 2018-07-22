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
NUM_CLASSES = 16
SEED = 1987

labels = pd.read_csv('dataset/kaggle/labels.csv')

selected_breed_list = list(labels.groupby('breed').count().sort_values(by = 'id', ascending = False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
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

# read image and conver to nparray
def read_img(img_id, set_type, size):
	img = image.load_img('dataset/kaggle/' + set_type + '/' + img_id + '.jpg', target_size = size)
	img = image.img_to_array(img)

	return img

model = ResNet50(weights = 'imagenet')

'''
j = int(np.sqrt(NUM_CLASSES))
i = int(np.ceil(1. * NUM_CLASSES / j))

fig = plt.figure(1, figsize = (16, 16))
grid = ImageGrid(fig, 111, nrows_ncols = (i, j), axes_pad = 0.05)

for i, (img_id, breed) in enumerate(labels.loc[labels['rank'] == 1, ['id', 'breed']].values):
	
	ax = grid[i]
	img = read_img(img_id, 'train', (224, 224))
	ax.imshow(img / 255.0)
	x = preprocess_input(np.expand_dims(img.copy(), axis = 0))
	predictions = model.predict(x)

	_, imagenet_class_name, probs = decode_predictions(predictions, top = 1)[0][0]
	ax.text(10, 180, 'ResNet50: %s (%.2f)' % (imagenet_class_name, probs), size = 7, color = 'w', backgroundcolor = 'k', alpha = 0.8)
	ax.text(0, 200, 'LABEL: %s' % breed, size = 7, color = 'k', backgroundcolor = 'w', alpha = 0.8)
	ax.axis('off')

plt.show()
'''
########################################
# VGG BOTTLENECK EXTRACTION AND LOGREG #
########################################

POOLING = 'avg'
'''
x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype = 'float32')

for i, img_id in tqdm(enumerate(labels['id'])):
	img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
	x = preprocess_input(np.expand_dims(img.copy(), axis = 0))
	x_train[i] = x

print('Train images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr = x_train[train_index]
Xv = x_train[valid_index]
print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))

vgg_bottleneck = VGG16(weights = 'imagenet', include_top = False, pooling = POOLING)
train_vgg_bf = vgg_bottleneck.predict(Xtr, batch_size = 32, verbose = 1)
valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size = 32, verbose = 1)

print('VGG train bottleneck features shape: {} size: {:,}'.format(train_vgg_bf.shape, train_vgg_bf.size))
print('VGG validation bottleneck features shape: {} size: {:,}'.format(valid_vgg_bf.shape, valid_vgg_bf.size))

# logistic regression on vgg bottleneck features
lr = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', random_state = SEED)
lr.fit(train_vgg_bf, (ytr * range(NUM_CLASSES)).sum(axis = 1))

valid_probs = lr.predict_proba(valid_vgg_bf)
valid_predictions = lr.predict(valid_vgg_bf)

print('Validation VGG Loss: {}'.format(log_loss(yv, valid_probs)))
print('Validation VGG Accuracy: {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis = 1), valid_predictions)))
'''
############################################
# XCEPTION BOTTLENCK EXTRACTION AND LOGREG #
############################################

INPUT_SIZE = 299
x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype = 'float32')

for i, img_id in tqdm(enumerate(labels['id'])):
	img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
	x = xception.preprocess_input(np.expand_dims(img.copy(), axis = 0))
	x_train[i] = x
	
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr = x_train[train_index]
Xv = x_train[valid_index]

print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))

xception_bottleneck = xception.Xception(weights = 'imagenet', include_top = False, pooling = POOLING)

train_x_bf = xception_bottleneck.predict(Xtr, batch_size = 32, verbose = 1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size = 32, verbose = 1)
print('Xception training bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
print('Xception validation bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))

