import pandas as pd
import numpy as np
import sklearn.linear_model as lm

songs_train = pd.read_csv("songs_train.csv")

songs_test = pd.read_csv("songs_test.csv")

X_train = songs_train.drop(["popularity"], axis=1)

y_train = songs_train["popularity"]

X_test = songs_test.drop(["popularity"], axis=1)

model = lm.LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

songs_test["popularity"] = y_pred

songs_test.to_csv("songs_testout.csv", index=False)
