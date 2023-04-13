import pickle
from collections import namedtuple

import implicit
import numpy as np
from p_at_k import precision_at_k_score
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sleec.ensemble import Model, Ensemble
from sleec.helpers import project
from tqdm import tqdm

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

binarized_labels = csr_matrix(mlb.transform(labels))

number_of_labels = len(mlb.classes_)

del mlb, labels, files

train_texts, test_texts, train_labels, test_labels, = train_test_split(texts, binarized_labels, test_size=.2,
                                                                       random_state=42)
size = (0.1 * binarized_labels.shape[0]) / train_labels.shape[0]
del texts, binarized_labels
train_texts, validation_texts, train_labels, validation_labels = train_test_split(list(train_texts), train_labels,
                                                                                  test_size=size, random_state=42)

train_embeddings = []
for i in ['1', '1.5', '2']:
    fileObj = open('embeddings_train' + str(i) + '.obj', 'rb')
    temp = pickle.load(fileObj)
    [train_embeddings.append(i) for i in temp]
    fileObj.close()
train_embeddings = np.array(train_embeddings)
fileObj = open('embeddings_test.obj', 'rb')
test_embeddings = np.array(pickle.load(fileObj))
fileObj.close()

del validation_texts, validation_labels

params = namedtuple('args', ['num_learner', 'num_clusters',
                             'num_threads', 'SVP_neigh', 'out_dim',
                             'w_thresh', 'sp_thresh', 'cost',
                             'NNtest', 'normalize'])

params.num_learners = 15  # 1
params.num_clusters = int(train_labels.shape[0] / 7000)  # 1
params.num_threads = 64
params.SVP_neigh = 50
params.out_Dim = 500  # 5000
params.w_thresh = 0.01  # ?
params.sp_thresh = 0.01  # ?
params.NNtest = 5
params.normalize = 1  # ?
params.regressor_lambda1 = 1e-6
params.regressor_lambda2 = 1e-3
params.embedding_lambda = 0.1  # determined automatically in WAltMin_asymm.m

clusterings = []
for i in range(params.num_learners):
    model = KMeans(n_clusters=params.num_clusters, n_init=8, max_iter=100)
    model.fit(train_embeddings)
    clusterings.append(model)

learners = []
for clus_model in tqdm(clusterings):
    models = []
    for i in range(clus_model.n_clusters):
        data_idx = np.nonzero(clus_model.labels_ == i)[0]
        X = train_embeddings[data_idx, :]
        Y = train_labels[data_idx, :]
        graph = kneighbors_graph(Y, params.SVP_neigh, mode='distance', metric='cosine',
                                 include_self=True,
                                 n_jobs=-1)
        graph.data = 1 - graph.data  # convert to similarity        
        als_model = implicit.als.AlternatingLeastSquares(factors=params.out_Dim,
                                                         regularization=params.embedding_lambda)
        als_model.fit(graph)
        Z = als_model.item_factors
        if True:
            regressor = ElasticNet(alpha=0.1, l1_ratio=0.001)
            regressor.fit(X, Z)
            V = regressor.coef_
        else:
            V = learn_V(X.toarray(), Z,
                        lambda1=params.regressor_lambda1,
                        lambda2=params.regressor_lambda2,
                        iter_max=200,
                        print_log=True)
        fitted_Z = X @ V.T
        Z_neighbors = NearestNeighbors(n_neighbors=params.NNtest, metric='cosine').fit(fitted_Z)
        projected_center = project(V, clus_model.cluster_centers_[i])
        learned = {
            'center_z': projected_center,
            'V': V,
            'Z_neighbors': Z_neighbors,
            'data_idx': data_idx
        }
        models.append(learned)
    learners.append(models)

models = [Model(learner, train_labels)
          for learner in learners]
ensemble = Ensemble(models)
predictions = ensemble.predict_many(test_embeddings)

new_predictions = []
for prediction in predictions:
    new_predictions.append([1 if i > 0 else 0 for i in prediction])

print(f1_score(test_labels, np.array(new_predictions), average='macro'))
print(precision_at_k_score(test_labels.A, new_predictions))
