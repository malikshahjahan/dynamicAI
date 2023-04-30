import pickle

import numpy as np

model = pickle.load(open('finalized_model.sav', 'rb'))
X_test = np.array([0,10.2,10.2,0.77,3.9284000000000003,152.0,14.9569,0.0,1015.51])
y_test = model.predict(X_test.reshape(-1, 9))
score = model.score(X_test.reshape(-1, 9), y_test)
print(score)
print(y_test)