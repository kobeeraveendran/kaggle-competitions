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

plt.show()