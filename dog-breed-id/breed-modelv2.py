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