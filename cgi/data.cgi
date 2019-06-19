#!/usr/bin/python3

import cgi, cgitb
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



# Create instance of FieldStorage
form = cgi.FieldStorage()

gender = form.getvalue('gender')
age = form.getvalue('age')
cp = form.getvalue('cp')
trestbps = form.getvalue('trestbps')
chol = form.getvalue('chol')
fbs = form.getvalue('fbs')
restecg = form.getvalue('restecg')
thalach = form.getvalue('thalach')
exang = form.getvalue('exang')

oldpeak = form.getvalue('oldpeak')

slope = form.getvalue('slope')

ca = form.getvalue('ca')

thal = form.getvalue('thal')



# machine learning code

df = pd.read_csv('/var/www/cgi-bin/heart.csv')


features = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
features = features.fillna(features.mean())
features = features.values
label = df[['target']].values



x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.01,random_state=4)

clf = RandomForestClassifier(criterion='entropy',n_estimators=7)


trained = clf.fit(x_train,y_train)

predicted = trained.predict(x_test)



from numpy import array
a = array( [age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal] )


a.reshape(-1,1)


result = list(trained.predict([a]))[0]



print("Content-type:text/html\r\n\r\n")
print('<html>')
print('<head>')
print('<title>Hello Word - First CGI Program</title>')
print('</head>')
print('<body>')
print('<h1>Heart disease predictor is here where are you??\n\n</h1>')
#print('<h2> predicted output is --:',result,'</h2>')
if result == 0:
  print("<h2> You dont have heart disease. </h2>")
else:
  print("<h2> Ohh! You may have heart disease. </h2>")
#print('<h3>accuracy from random forest algorithm',metrics.accuracy_score(y_test,predicted)*100,'</h3>')
#print('<h3>values</h3>')
print('</body>')
print('</html>')

