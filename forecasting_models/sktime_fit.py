from sktime.datasets import load_arrow_head
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.transformations.panel.rocket import Rocket

df = pd.read_csv('data_with_labels.csv')
n = 20000
x_raw = df['close'].tolist()[:n]
y_raw = df['output'].tolist()[:n]

scaler = StandardScaler()
x_raw = list(scaler.fit_transform(np.array(x_raw).reshape(-1, 1)))

time_offset = int(1440)

x_data = []
y_data = []
for i in range(time_offset, len(x_raw)):
    x_data.append(x_raw[i - time_offset:i])
    y_data.append(y_raw[i - time_offset])

x_data = np.array(x_data)
x_data = from_2d_array_to_nested(x_data.reshape(x_data.shape[0], x_data.shape[1]))
y_data = np.array(y_data)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data)

rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
X_train_transform = rocket.fit_transform(X_train)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_train_transform, y_train)

X_test_transform = rocket.transform(X_test)
classifier.score(X_test_transform, y_test)
