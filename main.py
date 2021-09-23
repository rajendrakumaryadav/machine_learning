import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("./Salary.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

regeressor = LinearRegression()
regeressor.fit(X_train, Y_train)

y_pred = regeressor.predict(X_test)

print(y_pred)

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regeressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (Training Data Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()