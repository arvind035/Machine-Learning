# for one feature
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
# print(diabetes.DESCR)
# print(diabetes.data)
# print(diabetes.target)
# print(diabetes.frame)
# print(diabetes.feature_names)
# print(diabetes.data_filename)
# print(diabetes.target_filename)

diabetes_x=diabetes.data[:,np.newaxis,2]
# print(diabetes_x)

diabetes_x_train= diabetes_x[:50]
diabetes_x_test=diabetes_x[-30:]

diabetes_y_train=diabetes.target[:50]
diabetes_y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)

diabetes_y_predict=model.predict(diabetes_x_test)

print("sse",mean_squared_error(diabetes_y_test,diabetes_y_predict))
print(("Weights",model.coef_))
print(("Intrecep",model.intercept_))
 # sse 3563.5259785449803
 # ('Weights', array([458.15762507]))
 # ('Intrecep', 144.70176785409583)

plt.scatter(diabetes_x_train,diabetes_y_train)
# plt.plot(diabetes_y_predict)
plt.plot(diabetes_x_test,diabetes_y_predict)
plt.show()

