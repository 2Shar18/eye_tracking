import pandas as pd
df = pd.read_csv('trainset.csv')
X = df[["leftEAR","xLStart","xLEnd","yLStart","yLEnd","xL","yL","wL","hL","xLPoint","yLPoint","rightEAR","xRStart","xREnd","yRStart","yREnd","xR","yR","wR","hR","xRPoint","yRPoint"]]
y = df[['yOut']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(x_train,y_train)

x_arr = []
predictions = lm.predict([x_train.iloc[0]])[0]
x_arr.append(predictions[0])
print(predictions)[0:5]

print(y_train)[0:5]

lm.score(x_test,y_test)

lm.coef_

lm.intercept_

import pickle
filename = 'trained_model_y.sav'
pickle.dump(model, open(filename, 'wb'))