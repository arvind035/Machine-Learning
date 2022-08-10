from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris_data=datasets.load_iris()

features=iris_data.data
labels=iris_data.target

# print(features[0],labels[0])
# print(iris_data.keys())
# print(iris_data.DESCR)

clf=KNeighborsClassifier()
clf.fit(features,labels)
predict_clf1=clf.predict([[2,5,6,4]])
predict_clf2=clf.predict([[1,1,1,1]])
print(predict_clf1)
print(predict_clf2)


