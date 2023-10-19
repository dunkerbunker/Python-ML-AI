# Gaussian

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("new_dataset.csv")

outlook_at=LabelEncoder()
Temp_at=LabelEncoder()
Hum_at=LabelEncoder()
win_at=LabelEncoder()

inputs=df.drop('Play',axis='columns')
target=df['Play']

inputs['outlook_n']= outlook_at.fit_transform(inputs['Outlook'])
inputs['Temp_n']= outlook_at.fit_transform(inputs['Temp'])
inputs['Hum_n']= outlook_at.fit_transform(inputs['Humidity'])
inputs['win_n']= outlook_at.fit_transform(inputs['Windy'])

inputs_n=inputs.drop(['Outlook','Temp','Humidity','Windy'],axis='columns')

classifier = GaussianNB()
classifier.fit(inputs_n,target)

classifier.score(inputs_n,target)

classifier.predict([[0,0,0,1]])