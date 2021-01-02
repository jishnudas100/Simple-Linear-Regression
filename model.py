import pandas as pd

#Reading data 
a = pd.read_csv("Salary_Data.csv")
x=a.iloc[:,:-1]
y=a.iloc[:,-1]

#Spliting of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Regression model training 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting output for test data set
y_pred=regressor.predict(X_test)

#Displaying the regression line predicted by the model 
import matplotlib.pyplot as plt 
plt.scatter(X_train,y_train,color="red")
plt.plot(X_test,y_pred,color="blue")

#Taking input from input 
inp=[int(input("Enter the total no. of years of experience"))]

inp1= pd.DataFrame({'abc':inp})

print(regressor.predict(inp1))

