import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("house_prices.csv")

print("Dataset Preview:\n", df.head())
X = df[['Rooms', 'Size']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
example_prediction = model.predict([[3, 1500]])
print("\nPredicted Price for 3-room, 1500 sqft house:", example_prediction[0])
