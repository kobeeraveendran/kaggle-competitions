import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import re
import sklearn
import seaborn as sns
import xgboost as xgb
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
passenger_id = test['PassengerId']

# feature engineering, data cleaning
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

full_data = [train, test]

#print(train.head())
#print('\n\n\n\n')
#print(test.head())

# from Sina's kernel on Kaggle

def remove_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)

    return ''

for dataset in full_data:
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
    
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_ct = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_ct)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    dataset['Title'] = dataset['Name'].apply(remove_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 
                       'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 
                       'Sir', 'Jonkeer', 'Dona'], 'Rare')
    
    
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    
    # map titles to integers based on gender and class
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)

    # map embarked locations to integers
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # map fares to integer categories
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # map ages into integer categories
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalFare'] = pd.cut(train['Fare'], 4)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

train = train.drop(labels = ['CategoricalAge', 'CategoricalFare', 'PassengerId', 'Name', 'SibSp', 'Cabin', 'Ticket'], axis = 1)
test = test.drop(labels = ['PassengerId', 'Name', 'SibSp', 'Cabin', 'Ticket'], axis = 1)


#print(train.head())
#print('\n\n\n\n')
#print(test.head())

colormap = plt.cm.RdBu
plt.figure(figsize = (14, 12))
plt.title('Pearson correlation of features', y = 1.05, size = 15)
sns.heatmap(data = train.astype(float).corr(), linewidths = 0.1, vmax = 1.0, 
            square = True, cmap = colormap, linecolor = 'white', annot = True)

# singular matrix e - debug
'''
graph = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked', u'FamilySize', u'Title']], 
                     hue = 'Survived', 
                     palette = 'seismic',
                     height = 1.2, 
                     diag_kind = 'kde', 
                     diag_kws = dict(shade = True), 
                     plot_kws = dict(s = 10))
graph.set(xticklabels = [])
'''
#plt.show()

# useful constants
train_size = train.shape[0]
test_size = test.shape[0]
SEED = 0
NUM_FOLDS = 5
kf = KFold(train_size ,n_folds = NUM_FOLDS, random_state = SEED)

# helper class for sklearn functions
class SklearnHelper(object):
    def __init__(self, clf, seed = 0, params = None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, train_data, train_labels):
        self.clf.fit(train_data, train_labels)

    def predict(self, data):
        return self.clf.predict(data)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

# out-of-fold predictions
def get_oof(clf, train_data, train_labels, test_data):
    oof_train = np.zeros((train_size,))
    oof_test = np.zeros((test_size,))
    oof_test_skf = np.empty((NUM_FOLDS, test_size))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = train_data[train_index]
        y_tr = train_labels[train_index]
        x_te = train_data[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(test_data)

    oof_test[:] = oof_test_skf.mean(axis = 0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# classifer parameters

# random forest
rf_params = {
    'n_jobs': -1, # use all available cores
    'n_estimators': 500, # number of clf trees
    'warm_start': True, 
    'max_depth': 6, 
    'min_samples_leaf': 2, 
    'max_features': 'sqrt', 
    'verbose': 0
}

# extra trees
et_params = {
    'n_jobs': -1, 
    'n_estimators': 500, 
    'max_depth': 8, 
    'min_samples_leaf': 2, 
    'verbose': 0
}

# adaboost
ada_params = {
    'n_estimators': 500, 
    'learning_rate': 0.75
}

# gradient boosting
gb_params = {
    'n_estimators': 500, 
    'max_depth': 5, 
    'min_samples_leaf': 2, 
    'verbose': 0
}

# SVM
svm_params = {
    'kernel': 'linear', 
    'C': 0.025
}

# create objects for each model
# random forest, extra tree, adaboost, gradient boosting, svm

random_forest = SklearnHelper(clf = RandomForestClassifier, 
                   seed = SEED, 
                   params = rf_params)

extra_tree = SklearnHelper(clf = ExtraTreesClassifier, 
                           seed = SEED, 
                           params = et_params)

adaboost = SklearnHelper(clf = AdaBoostClassifier, 
                         seed = SEED, 
                         params = ada_params)

gradient_boosting = SklearnHelper(clf = GradientBoostingClassifier, 
                                seed = SEED, 
                                params = gb_params)

svm = SklearnHelper(clf = SVC, 
                    seed = SEED, 
                    params = svm_params)

# convert sets from dataframes to np arrays
train_labels = train['Survived'].ravel()
train = train.drop(['Survived'], axis = 1)
train_data = train.values
test_data = test.values


# first-level predicitions
et_oof_train, et_oof_test = get_oof(extra_tree, 
                                    train_data, 
                                    train_labels, 
                                    test_data)

print('Extra trees clf training complete')

rf_oof_train, rf_oof_test = get_oof(random_forest, 
                                    train_data, 
                                    train_labels, 
                                    test_data)

print('Random forest clf training complete')

ada_oof_train, ada_oof_test = get_oof(adaboost, 
                                      train_data, 
                                      train_labels, 
                                      test_data)

print('AdaBoost clf training complete')

gb_oof_train, gb_oof_test = get_oof(gradient_boosting, 
                                    train_data, 
                                    train_labels, 
                                    test_data)

print('Gradient boosting clf training complete')

svm_oof_train, svm_oof_test = get_oof(svm, 
                                      train_data, 
                                      train_labels, 
                                      test_data)

print('SVM training complete')

print('Level 1 Training complete')