# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MOHAMMED HAMZA M
RegisterNumber: 212224230167 
*/
```
```py
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```
## Output:
## DATA HEAD:
![WhatsApp Image 2025-04-21 at 10 18 17_8c0e8326](https://github.com/user-attachments/assets/6a5ecd49-f2f8-4176-8571-8b0dc17447ca)

## DATASET INFO:
![WhatsApp Image 2025-04-21 at 10 18 43_21852047](https://github.com/user-attachments/assets/e2aae17f-316a-46b9-a23b-30febfa95302)

## NULL DATASET:
![WhatsApp Image 2025-04-21 at 10 19 08_77ab6e66](https://github.com/user-attachments/assets/78dff978-a79f-4501-827b-1e9f9bff8a8d)

## VALUES COUNT IN THE LEFT COLUMN
![WhatsApp Image 2025-04-21 at 10 19 38_aa14e95f](https://github.com/user-attachments/assets/22ad8d34-0079-4379-855b-915bb2b9e924)

## DATASET TRANSFORMED HEAD:
![WhatsApp Image 2025-04-21 at 10 20 04_985dad20](https://github.com/user-attachments/assets/48470431-de6b-4dff-b82e-274dbe1f3959)

## X.HEAD:
![WhatsApp Image 2025-04-21 at 10 20 30_7aea0899](https://github.com/user-attachments/assets/75890159-2844-4801-a14c-78810989da1e)

## ACCURACY:
![WhatsApp Image 2025-04-21 at 10 20 58_8d31fc90](https://github.com/user-attachments/assets/dd6108e5-d767-4a62-aec5-171af029b8ed)

## DATA PREDICTION:
![WhatsApp Image 2025-04-21 at 10 21 22_65f93cd0](https://github.com/user-attachments/assets/5d217669-6247-4565-b923-c853016d79e8)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
