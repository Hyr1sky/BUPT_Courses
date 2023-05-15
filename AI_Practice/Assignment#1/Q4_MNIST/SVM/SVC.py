import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=load_digits()
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.3,random_state=42)

svc=SVC().fit(x_train,y_train)
d=svc.predict(x_test)
acc=svc.score(x_test,y_test)

print("The accuracy is %.3f" % acc)
y_predict=svc.predict(x_test)
print(y_predict)