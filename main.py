import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
T_data = pd.read_csv("train-data.csv")
V_data = pd.read_csv("test-data.csv")


def split(x):
    return (x.loc[:,["Kilometers_Driven","Mileage","Engine","Power"]], x.iloc[:, -1])


X_train, Y_train = split(T_data)
X_test, Y_test = split(V_data)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print(mean_squared_error(Y_pred,Y_test))