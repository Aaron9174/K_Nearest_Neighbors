# -------------------- #
# Title: K-Nearest-Neighbors
# Author: Aaron Hebson
# Acknowledgements/Credit:
# - O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear 
#     programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.
# - William H. Wolberg and O.L. Mangasarian: "Multisurface method of 
#     pattern separation for medical diagnosis applied to breast cytology", 
#     Proceedings of the National Academy of Sciences, U.S.A., Volume 87, 
#     December 1990, pp 9193-9196.
# - O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition 
#     via linear programming: Theory and application to medical diagnosis", 
#     in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying
#     Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.
# - K. P. Bennett & O. L. Mangasarian: "Robust linear programming 
#     discrimination of two linearly inseparable sets", Optimization Methods
#     and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).
# - Sentdex from YT

import os
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

#my_path = os.path.abspath(os.path.dirname("breast-cancer-wisconsin.data"))
df = pd.read_csv("../breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)
X = np.array(df.drop(["class"], 1))
Y = np.array(df["class"])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
