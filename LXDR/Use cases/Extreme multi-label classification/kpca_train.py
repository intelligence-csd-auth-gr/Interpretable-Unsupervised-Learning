import pickle

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

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

del texts, binarized_labels

number_of_reduced = 500  # try with 5000 as well

kpca = KernelPCA(kernel='rbf', n_components=number_of_reduced, random_state=42)
kpca.fit(train_labels)
fileObj = open('kpca_' + str(number_of_reduced) + '.obj', 'wb')
pickle.dump(kpca, fileObj)
fileObj.close()
