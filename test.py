import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/Lakshya Pawar/Desktop/academics/celebal technologies internship/week 5/assignment 5 (Statistics &Data)/train.csv")
print(df.shape)

cols = ['Name', 'Ticket', 'Cabin']
df = df.drop(cols, axis=1)
print(df.shape)

dummies = []
cols = ['Pclass', 'Sex', 'Embarked']
for col in cols:
    dummies.append(pd.get_dummies(df[col]).astype(float))

titanic_dummies = pd.concat(dummies, axis=1)
print(titanic_dummies.head(5))
print(titanic_dummies.dtypes)

df = pd.concat((df, titanic_dummies), axis=1)
df = df.drop(cols, axis=1)
print(df.dtypes)

df['Age'] = df['Age'].interpolate()
print(df['Age'].shape)

X = df.values
y = df['Survived'].values

X = np.delete(X, 1, axis=1)

print(df.info)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train)
print(X_test)

st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)
print(X_train)
print(X_test)