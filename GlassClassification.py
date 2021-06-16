#!/usr/bin/env python
# coding: utf-8


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


path = 'glass.csv'
df = pd.read_csv(path)
df.head()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#get_ipython().run_line_magic('matplotlib', 'inline')


df.info() ## No null value



df['Type'].value_counts()



## Gives correlation of different parameters on target class
df.corr() 



## checking for null values
df.isnull().sum()  



sns.pairplot(df, hue = 'Type')



plt.figure(figsize = (12,10))
sns.heatmap(df.corr(), annot = True, cmap = 'Greens')



cor = df.corr()['Type'].sort_values(ascending = False)
cor



plt.figure(figsize = (12,6))
cor.plot(kind = 'bar')


# Train test split


df.columns



X = df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = df['Type']
X



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# **Logical regression Model**


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)



pred1 = lg.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report


print(confusion_matrix(y_test, pred1))

print(classification_report(y_test, pred1))


# Since this is multi class classification let us try other models


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



y = df['Type']
X = pd.DataFrame(X, columns = df.columns[:-1])
X



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# **KNN Model**


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred2 = knn.predict(X_test)



print(confusion_matrix(y_test, pred2))



print(classification_report(y_test, pred2))


# We can observe some improvements over the previous results


# for loop to predict values for different neighbours value
error = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))



plt.figure(figsize = (10,6))
plt.plot(range(1,30), error, color = 'green', marker = 'o', markerfacecolor = 'blue')
plt.title('Error rate vs K value')
plt.xlabel('K-values')
plt.ylabel('Error')


# Here the error is increasing hence k = 1 is best choice

# **Decesion Tree**


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred3 = dtree.predict(X_test)



print(confusion_matrix(y_test, pred3))



print(classification_report(y_test, pred3))


# **Random Forest**


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred4 = rfc.predict(X_test)



print(confusion_matrix(y_test, pred4))

print(classification_report(y_test, pred4))



Accuracy = [66,57,62,80]
Models = ['Logical Regression','K-means clustor','Decesion Tree','Random Forest']



plt.figure(figsize = (8,4))
plt.bar(Models, Accuracy, color = 'maroon')
plt.title('Models vs Accuracy')





