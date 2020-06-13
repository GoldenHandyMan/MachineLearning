# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:03:30 2020

@author: filif
"""
def getTitle(name):
    comma_index = name.index(',')
    dot_index = name.index('.')
    return name[comma_index + 2: dot_index]


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("C:\\Users\\filif\\Desktop\\umcs.ai\\titanic.csv")
# dataset_1 = pd.read_csv("C:\\Users\\filif\\Desktop\\umcs.ai\\titanic.csv")
dataset = dataset.drop(['PassengerId', 'Ticket'], axis=1)


dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())


#biore tytuł osoby
temp = 0
for a in dataset['Name']:
    dataset.loc[temp, 'Name'] = getTitle(a)
    temp += 1

#biore numer pokładu
temp1 = 0
for a in dataset['Cabin']:
    dataset.loc[temp1, 'Cabin'] = str(a)[0]
    temp1 += 1
    

    
standardlist = ['Age', 'Fare']
labellist = ['Name', 'Sex', 'SibSp', 'Cabin', 'Embarked']

temp2 = 0
for a in dataset['Name']:
    if a != 'Mr' and a != 'Miss' and a != 'Mrs':
        dataset.loc[temp2, 'Name'] = 'Oth'
    temp2 += 1
        
sc = StandardScaler()
le = LabelEncoder()

dataset[standardlist] = sc.fit_transform(dataset[standardlist])

dataset = dataset.dropna()


for a in labellist:
    dataset[a] = le.fit_transform(dataset[a])

dataset[labellist] = pd.get_dummies(dataset[labellist])

X = dataset.iloc[:, 1:10]
y = dataset['Survived']


from sklearn.linear_model import LogisticRegression # 0.83
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # 0.82
from sklearn.neighbors import KNeighborsClassifier # 0.87
from sklearn.ensemble import RandomForestClassifier # 0.978
from sklearn.tree import DecisionTreeClassifier #0.97 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 12, test_size=0.2)

# model = LogisticRegression().fit(X, y)
# model = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0).fit(X, y)
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0).fit(X, y)
# print(model.score(X_test, y_test))
    
# from keras.layers import Dense
# from keras import Sequential

# classifier = Sequential()
# classifier.add(Dense(30, activation='relu', kernel_initializer='random_normal', input_dim=9))
# classifier.add(Dense(30, activation='relu', kernel_initializer='random_normal'))
# classifier.add(Dense(30, activation='relu',  kernel_initializer='random_normal'))
# classifier.add(Dense(30, activation='relu',  kernel_initializer='random_normal'))
# classifier.add(Dense(30, activation='relu',  kernel_initializer='random_normal'))
# classifier.add(Dense(1, activation='sigmoid',  kernel_initializer='random_normal'))
# classifier.compile(optimizer = 'adam',loss = 'mse', metrics = ['acc'])

# classifier.fit(X_train, y_train, epochs = 500, verbose = 1)

print(model.score(X_test, y_test))














