import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import feature_phase1_functions

df = pd.read_csv("C:/Users/Lakshya Pawar/Desktop/academics/celebal technologies internship/week 5/assignment 5 (Statistics &Data)/titanic.csv")
print(df.shape)

print(df.isnull().sum())

age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Pclass'])
df['Pclass_1'] = df['Pclass_1'].astype(int)
df['Pclass_2'] = df['Pclass_2'].astype(int)
df['Pclass_3'] = df['Pclass_3'].astype(int)

df = pd.get_dummies(df, columns=['Embarked'])
le = LabelEncoder()
df['Embarked_C'] = le.fit_transform(df['Embarked_C'])
df['Embarked_Q'] = le.fit_transform(df['Embarked_Q'])
df['Embarked_S'] = le.fit_transform(df['Embarked_S'])

cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
df = df.drop(cols, axis=1)
print(df.shape)

print(df.isnull().sum())
print(df.dtypes)
print(df.head(25))

cols = ['Age', 'Fare']
for col in cols:
    feature_phase1_functions.replace_with_thresholds(df, col)
cols = df.columns
for col in cols:
    print(col, feature_phase1_functions.check_outlier(df, col))

print(df.isnull().sum())
print(df.head(25))

X = df.drop(columns=['Survived']) # feature dataset
y = df['Survived'] # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.head(10))

scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

print(X_train.head(10))
