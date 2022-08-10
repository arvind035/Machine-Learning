#logistics for one element(feature)
# print 1 that means virginica if 0 thatmeans not virginica
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()
feature=iris["data"][:,3:]
label=(iris["target"] == 2).astype(np.int)

clf=LogisticRegression()
clf.fit(feature,label)
output1=clf.predict([[1.6]])
output2=clf.predict([[2.6]])
print(output1)
print(output2)