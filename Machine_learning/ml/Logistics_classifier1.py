#logistics for multiple element(feature)

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()
feature=iris.data
label=iris.target

clf=LogisticRegression()
clf.fit(feature,label)
output1=clf.predict([[3.6,1.5,4.9,5.6]])
output2=clf.predict([[1.3,1.9,2.5,1.7]])
print(output1)
print(output2)