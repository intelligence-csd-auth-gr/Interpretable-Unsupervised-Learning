import pickle

import numpy as np
from p_at_k import precision_at_k_score
from sklearn.metrics import f1_score

number_of_reduced = 500
fileObj = open('predictions_of_exml_' + str(number_of_reduced) + '.obj', 'rb')
red = pickle.load(fileObj)
fileObj.close()

predictions = red[0]
y_test = red[1]
new_predictions = []
for prediction in predictions:
    for pred in prediction:
        new_predictions.append([1 if i > 0.5 else 0 for i in pred])

print(f1_score(np.array(y_test), np.array(new_predictions), average='macro'))

new_predictions = []
for prediction in predictions:
    for pred in prediction:
        new_predictions.append(pred)
predictions = new_predictions

print(precision_at_k_score(y_test, predictions, 1, 1))
