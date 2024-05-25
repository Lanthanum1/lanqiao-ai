import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


train = pd.read_csv('songs_train.csv')
X = train.drop(['popularity'], axis=1)
y = train['popularity'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"{r2:.2f}")    

test = pd.read_csv('songs_test.csv')
X_test = test.drop(['popularity'], axis=1)
predictions = model.predict(X_test)
test["popularity"] = predictions
test.to_csv('songs_testout.csv', index=False)