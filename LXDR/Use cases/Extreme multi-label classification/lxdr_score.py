import pickle

import numpy as np
from joblib import Parallel, delayed
from p_at_k import precision_at_k_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from lxdr import LXDR

files = open('mesh_2020_test.txt')
texts = []
labels = []
counter = 0
for line in files:
    texts.append(line[2:-2].split('abstracttt: ')[1].split('labels: ')[0])
    labels.append(line[2:-2].split('abstracttt: ')[1].split('labels: ')[1][1:].split('#'))
files.close()

mlb = MultiLabelBinarizer()
mlb.fit(labels)

binarized_labels = mlb.transform(labels)

number_of_labels = len(mlb.classes_)

del mlb, labels, files

train_texts, test_texts, train_labels, test_labels, = train_test_split(texts, binarized_labels, test_size=.2,
                                                                       random_state=42)

size = (0.1 * len(binarized_labels)) / len(train_labels)
train_texts, validation_texts, train_labels, validation_labels = train_test_split(list(train_texts), train_labels,
                                                                                  test_size=size, random_state=42)

del texts, train_texts, test_texts, validation_texts, binarized_labels

number_of_reduced = 500
fileObj = open('kpca_' + str(number_of_reduced) + '.obj', 'rb')
kpca = pickle.load(fileObj)
fileObj.close()

fileObj = open('predicted_instances_kpca_' + str(number_of_reduced) + '.obj', 'rb')
result = pickle.load(fileObj)
mms = result[1]
predicted_reduced_test = mms.inverse_transform(result[0])
fileObj.close()
del result

ours = LXDR(kpca, ['L' + str(i) for i in range(17584)], train_labels[:10], False, ltype='locallocal', n_jobs=1)
del kpca

y_test = test_labels

ours.initial_data = train_labels
del train_labels
ours._set_knn_latent_local()

number_of_neighbs = 5
bob = 0


def bob2(idd):
    instance = predicted_reduced_test[idd]
    components_ = ours.explain_reduced_instance(instance, number_of_neighbours=number_of_neighbs, auto_alpha=False,
                                                ng_technique='LatentKNN')
    return np.dot(instance, components_)


predictions = Parallel(n_jobs=50, verbose=1)(
    delayed(bob2)(idd) for idd in tqdm(range(bob, len(predicted_reduced_test))))

new_predictions = []
for prediction in predictions:
    new_predictions.append([1 if i > 0 else 0 for i in prediction])

print(f1_score(np.array(y_test), np.array(new_predictions), average='macro'))

new_predictions = []
for prediction in predictions:
    new_predictions.append([1 if i > 0.5 else 0 for i in prediction])

print(f1_score(np.array(y_test), np.array(new_predictions), average='macro'))

precision_at_k_score(y_test, predictions)
