import numpy as np
from sklearn import svm
from sklearn.datasets  import load_digits
from sklearn.model_selection  import train_test_split
# import _pickle as pickle

if __name__ == '__main__':
    mnist = load_digits()
    x,test_x,y,test_y = train_test_split(mnist.data, mnist.target, test_size = 0.25, random_state = 40)
    
    model = svm.LinearSVC()
    model.fit(x, y)

    z = model.predict(test_x)

    print('准确率:',np.sum(z == test_y)/z.size)

#   with open('./model.pkl','wb') as file:
#       pickle.dump(model,file)

"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import _pickle as pickle

data=load_digits()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

svc=SVC().fit(x_train,y_train)
d=svc.predict(x_test)

acc=svc.score(x_test,y_test)

print("The accuracy is %.3f" % acc)
y_predict=svc.predict(x_test)
print(y_predict)
"""