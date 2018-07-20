import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
passenger_id = test['PassengerId']

# feature engineering, data cleaning

train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

full_data = [train, test]

print(train.head())
print('\n\n\n\n')
print(test.head())

# from Sina's kernel on Kaggle
for dataset in full_data:
    dataset['FamilySize'] = dataset['Parch'] + dataset['Sibsp'] + 1
    
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_ct = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_ct)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['age'] = dataset['Age'].astype(int)

train['CategoricalFare'] = pd.cut(train['Fare'], 4)
train['CategoricalAge'] = pd.cut(train['Age'], 5)


train = train.drop(labels = ['Name', 'SibSp', 'Ticket'], axis = 1)