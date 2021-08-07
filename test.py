import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train(keys,values):

	ascii_values = [ord(c) for c in keys]
	values = np.array(values).reshape(len(values),1)
	ascii_values = np.array(ascii_values).reshape(len(ascii_values),1)
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)

	return regressor

def test(test_string,regressor):
	val = regressor.predict(np.array(ord(test_string)).reshape(-1,1))
	if val > 0.5:
		return 1
	else :
		return 0








