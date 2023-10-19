import pandas as pd
from sklearn.model_selection import train_test_split

# read pdf
df = pd.read_csv('D:\Projects\Python ML\churn model\Churn.csv')

X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

y_train.head()

# print(X_train)

from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# convert to numpy arrays || convert to float32 because one of the columns is bool
X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)  
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

model.fit(X_train, y_train, epochs=200, batch_size=32)

y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

accuracy_score(y_test, y_hat)

model.save('tfmodel')


new_test_df = pd.read_csv('D:\Projects\Python ML\churn model\Test.csv')

print(new_test_df)

new_X_test = pd.get_dummies(new_test_df.drop(['Churn', 'Customer ID'], axis=1))

print(new_X_test)

model = load_model('tfmodel')

new_X_test = np.asarray(new_X_test).astype(np.float32)

new_y_hat = model.predict(new_X_test)
new_y_hat = [0 if val < 0.5 else 1 for val in new_y_hat]

new_y_hat