import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.backend as K
from joblib import Parallel, delayed
from sklearn.cluster import Birch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class LXDR:
    """Local Explanation of Dimensionality Reduction (LXDR)

    A technique for interpreting locally the results of DR techniques.
    
    """

    def __init__(self,
                 dimensionality_reduction_model,
                 feature_names,
                 initial_data=None,
                 mean=None,
                 ltype='neural',
                 n_jobs=1):
        """Init function.

        Args:
        dimensionality_reduction_model: model used to dimensionally reduce 
                    the data (should be fitted on the initial dataset prior the 
                    object creation).
        feature_names: The names of the dataset column
        scope: If the techinque will work locally or globally. Accepted values 'local', or 'global'.
        initial_data: 2d array contaning the orgina dataset
        mean: if mean is going to be used during the training of the local models
        type: classic= 1 ridge model per dimension with shared neighbours, neural= 1 model for all dimensions with shared neighbours, local_local= 1 model per dimension with different neighbours per dimension
        """

        self.model = dimensionality_reduction_model
        self.number_of_reduced_dims = len(self._pseudo_transform(initial_data[:2])[0])

        self.feature_names = feature_names
        self.initial_data = initial_data

        self.n_jobs = n_jobs
        if self.n_jobs == 1:
            self._set_knn()
            self._set_knn_local()
            self._set_knn_latent()
            self._set_knn_latent_local()
        else:
            self._set_knn_local()

        # Not needed for text
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.zeros(initial_data[0].shape)
        self.global_ = {}
        self.birch = {}
        self.type = ltype

    def _pseudo_transform(self, input_data):
        if str(type(self.model)) == "<class '__main__.Autoencoder'>":
            return self.model.predict(np.array(input_data))
        else:
            return self.model.transform(input_data)

    def _set_knn(self):
        self.knn = NearestNeighbors()
        self.knn.fit(self.initial_data)

    def _set_knn_local(self):
        if self.n_jobs > 1:
            self.temp_data = self._pseudo_transform(self.initial_data)
        else:
            self.knn_local = {}
            self.temp_data = self._pseudo_transform(self.initial_data)
            temp_data = self._pseudo_transform(self.initial_data)
            for i in range(0, self.number_of_reduced_dims):
                enriched_data = np.hstack((self.initial_data, temp_data[:, i].reshape((len(temp_data[:, i]), 1))))
                self.knn_local[i] = NearestNeighbors().fit(enriched_data)

    def _set_knn_latent(self):
        self.knn_latent = NearestNeighbors()
        self.knn_latent.fit(self._pseudo_transform(self.initial_data))

    def _set_knn_latent_local(self):
        self.knn_latent_local = {}
        temp_data = self._pseudo_transform(self.initial_data)
        for i in range(0, self.number_of_reduced_dims):
            self.knn_latent_local[i] = NearestNeighbors().fit(temp_data[:, i].reshape((len(temp_data[:, i]), 1)))

    def best_alpha(self, X, y, distances):
        """ Finds the best linear model for our data by searching multiple values 
        for the regularization strength (alpha), 

        The peformance is determined by the mean_squared_error between the predicted
        and the true values.

        Args:
          X: 2d array of shape (# neigbors, features)

          y: 2d array of shape (# neighbors, reduced_features)

          distances: 1d array contaning the distances between the examined instance
              and its neighbors

        Returns:
          best_model: the best permorming Ridge model 
        """

        best_model = None
        best_score = 1000000000
        alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 1.9, 10, 100, 1000]
        for a in alphas:
            temp_model = Ridge(alpha=a, fit_intercept=False, random_state=7).fit(X, y)
            temp_perfomance = mae(y, temp_model.predict(X), sample_weight=distances)
            if temp_perfomance < best_score:
                best_score = temp_perfomance
                best_model = temp_model
        return best_model

    def _choose_number_of_neighbours(self, instance):
        """ Simpel funtion determining the number of neighbors in case it is not defined 
          by the user.

          Returns:
            int: number of neighbors
        """
        return int(3 * len(self.initial_data) / 4)  # More than number of features or equal

    def _get_coef(self, neighbours, transformed, distances=None, auto_alpha=False):
        """ Generates the components (weights) of LXDR

        Args:
          neighbours: 2d array of shape (# neigbors, features) with the neighbors

          transformed: 2d array of shape (# neighbors, reduced_features), the reduced 
              representation of the neighbors

          distances: 1d array contaning the distances between the examined instance
              and its neighbors. If distances are None, this is the Global Variant of LXDR (GXDR)

          auto_alpha: if True, function best_alpha will be used to find most 
              suitable alpha for the linear model. If False, default value of
              alpha will be used (alpha=1.0).

        Returns:
          array: 2d array of shape (n_components, n_features)
              containing weights where n_components is the dimension of the 
              reduced space, and n_features is the number of dimensions of 
              the original space.
        """
        reduced_dimensions = self.number_of_reduced_dims  # len(transformed[0])
        _coef_per_component = []

        if self.type == 'neural':
            input_layer = tf.keras.Input((len(neighbours[0]),))
            hidden_layer = tf.keras.layers.Dense(len(neighbours[0]), activation='relu', use_bias=False)(input_layer)
            output_layer = tf.keras.layers.Dense(reduced_dimensions, activation='linear', use_bias=False)(hidden_layer)
            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=0,
                                                                       restore_best_weights=True)
            adam = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=adam, loss='mse')

            if distances is not None:
                model.fit(neighbours - self.mean, transformed, sample_weight=distances,
                          epochs=1000, callbacks=[early_stopping_callback],
                          verbose=0)
            else:
                if self.global_ == {}:
                    model.fit(neighbours - self.mean, transformed,
                              epochs=1000, callbacks=[early_stopping_callback],
                              verbose=0)
                    self.global_['model'] = model
                else:
                    model = self.global_['model']
            _coef_per_component = ig(neighbours[0] - self.mean, model, reduced_dimensions)
        elif self.type == 'classic':
            if self.n_jobs != 1:
                _coef_per_component = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._auto_alpha)(neighbours - self.mean, transformed[:, cp], distances) for cp in
                    range(reduced_dimensions))
            else:
                for component in range(reduced_dimensions):
                    tc = transformed[:, component]
                    if auto_alpha:
                        linear_model = self.best_alpha(neighbours - self.mean, tc, distances)
                    else:
                        if distances is not None:
                            linear_model = Ridge(alpha=1.0, fit_intercept=False, random_state=7).fit(
                                neighbours - self.mean, tc, distances)
                        else:  # is Global!
                            if component not in self.global_:
                                linear_model = Ridge(alpha=1.0, fit_intercept=False, random_state=7).fit(
                                    neighbours - self.mean, tc)
                                self.global_[component] = linear_model
                            else:
                                linear_model = self.global_[component]
                    _coef_per_component.append(linear_model.coef_)
        else:  # localocal
            if self.n_jobs != 1:
                _coef_per_component = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._auto_alpha)(neighbours[cp] - self.mean, transformed[cp][:, cp], distances[cp]) for cp
                    in range(reduced_dimensions))
            else:
                for component in range(reduced_dimensions):
                    tc = transformed[component][:, component]
                    if auto_alpha:
                        linear_model = self.best_alpha(neighbours[component] - self.mean, tc, distances[component])
                    else:
                        linear_model = Ridge(alpha=1.0, fit_intercept=False, random_state=7).fit(
                            neighbours[component] - self.mean, tc, distances[component])

                    _coef_per_component.append(linear_model.coef_)
        return np.array(_coef_per_component)

    def _auto_alpha(self, neighbours, transformed_component, distances):
        return self.best_alpha(neighbours, transformed_component, distances).coef_

    def _ng_knn(self, instance, number_of_neighbours):
        if self.type != 'locallocal':
            neighbours = []
            neighbours_ind = self.knn.kneighbors([instance],
                                                 number_of_neighbours,
                                                 return_distance=False)[0]
            for i in neighbours_ind:
                neighbours.append(self.initial_data[i])
            neighbours = np.array(neighbours)
            neighbours = np.concatenate((np.array([instance]),
                                         neighbours))  # when we can't get the reduced dimensions for the instance's embeddings
            return neighbours

        else:
            transformed_instance = self._pseudo_transform([instance, instance])[0]
            neighbours_per_dim = {}
            for i in range(0, self.number_of_reduced_dims):
                new_instance = list(instance).copy()
                new_instance.append(transformed_instance[i])

                if self.n_jobs > 1:
                    enriched_data = np.hstack(
                        (self.initial_data, self.temp_data[:, i].reshape((len(self.temp_data[:, i]), 1))))
                    knn_local = NearestNeighbors().fit(enriched_data)
                    neighbours_ind = knn_local.kneighbors(np.array([new_instance]).reshape(1, -1),
                                                          number_of_neighbours,
                                                          return_distance=False)[0]
                else:
                    neighbours_ind = self.knn_local[i].kneighbors(np.array([new_instance]).reshape(1, -1),
                                                                  number_of_neighbours,
                                                                  return_distance=False)[0]
                neighbours = list()
                for j in neighbours_ind:
                    neighbours.append(self.initial_data[j])
                neighbours_per_dim[i] = np.concatenate((np.array([instance]), np.array(neighbours)))
            return neighbours_per_dim

    def _ng_latent_knn_inverse(self, instance, number_of_neighbours):
        transformed_instance = instance
        if self.type != 'locallocal':
            neighbours = []
            neighbours_ind = self.knn_latent.kneighbors([transformed_instance],
                                                        number_of_neighbours,
                                                        return_distance=False)[0]
            for i in neighbours_ind:
                neighbours.append(self.initial_data[i])
            neighbours = np.array(neighbours)
            return neighbours
        else:
            neighbours_per_dim = {}
            for i in range(0, self.number_of_reduced_dims):
                neighbours_ind = self.knn_latent_local[i].kneighbors([[transformed_instance[i]]],
                                                                     number_of_neighbours,
                                                                     return_distance=False)[0]
                neighbours = list()
                for j in neighbours_ind:
                    neighbours.append(self.initial_data[j])
                neighbours_per_dim[i] = np.array(neighbours)
            return neighbours_per_dim

    def _ng_latent_knn(self, instance, number_of_neighbours):
        transformed_instance = self._pseudo_transform([instance, instance])[0]
        if self.type != 'locallocal':
            neighbours = []
            neighbours_ind = self.knn_latent.kneighbors([transformed_instance],
                                                        number_of_neighbours,
                                                        return_distance=False)[0]
            for i in neighbours_ind:
                neighbours.append(self.initial_data[i])
            neighbours = np.array(neighbours)
            neighbours = np.concatenate((np.array([instance]),
                                         neighbours))  # when we can't get the reduced dimensions for the instance's embeddings
            return neighbours
        else:
            neighbours_per_dim = {}
            for i in range(0, self.number_of_reduced_dims):
                neighbours_ind = self.knn_latent_local[i].kneighbors([[transformed_instance[i]]],
                                                                     number_of_neighbours,
                                                                     return_distance=False)[0]
                neighbours = list()
                for j in neighbours_ind:
                    neighbours.append(self.initial_data[j])
                neighbours_per_dim[i] = np.concatenate((np.array([instance]), np.array(neighbours)))
            return neighbours_per_dim

    def _ng_clustering(self, instance, number_of_neighbours):
        transformed_instance = self._pseudo_transform([instance, instance])[0]
        number_of_clusters = int(len(self.initial_data) / number_of_neighbours)
        if self.type != 'locallocal':
            copy_of_initial = self.initial_data.copy()
            if number_of_clusters not in self.birch:
                clustering = Birch(
                    n_clusters=number_of_clusters).fit(copy_of_initial)
                x = clustering.labels_
                clusters = {}
                for i in set(x):
                    clusters[i] = []
                for i in range(len(x)):
                    clusters[x[i]].append(i)
                self.birch[number_of_clusters] = [clusters, clustering]
            else:
                clusters, clustering = self.birch[number_of_clusters]
            label = clustering.predict([instance])[0]
            neighbours = []
            for i in clusters[label]:
                neighbours.append(self.initial_data[i])

            neighbours = np.array(neighbours)
            neighbours = np.concatenate((np.array([instance]),
                                         neighbours))  # when we can't get the reduced dimensions for the instance's embeddings
            return neighbours
        else:
            neighbours_per_dim = {}

            temp_data = self._pseudo_transform(self.initial_data)
            for j in range(0, self.number_of_reduced_dims):
                new_instance = list(instance).copy()
                new_instance.append(transformed_instance[j])
                enriched_data = np.hstack((self.initial_data, temp_data[:, j].reshape((len(temp_data[:, j]), 1))))
                temp_list = list(enriched_data)
                if tuple([number_of_clusters, j]) not in self.birch:
                    clustering = Birch(
                        n_clusters=number_of_clusters).fit(temp_list)
                    x = clustering.labels_
                    clusters = {}
                    for i in set(x):
                        clusters[i] = []
                    for i in range(len(x)):
                        clusters[x[i]].append(i)
                    self.birch[tuple([number_of_clusters, j])] = [clusters, clustering]
                else:
                    clusters, clustering = self.birch[tuple([number_of_clusters, j])]
                label = clustering.predict([new_instance])[0]
                neighbours = []
                for i in clusters[label]:
                    neighbours.append(self.initial_data[i])
                neighbours = np.array(neighbours)
                neighbours_per_dim[j] = np.concatenate((np.array([instance]), np.array(neighbours)))
            return neighbours_per_dim

    def _ng_latent_clustering(self, instance, number_of_neighbours):
        transformed_instance = self._pseudo_transform([instance, instance])[0]
        number_of_clusters = int(len(self.initial_data) / number_of_neighbours)
        copy_of_initial = self._pseudo_transform(self.initial_data.copy())
        if self.type != 'locallocal':
            if number_of_clusters not in self.birch:
                clustering = Birch(
                    n_clusters=number_of_clusters).fit(copy_of_initial)
                x = clustering.labels_
                clusters = {}
                for i in set(x):
                    clusters[i] = []
                for i in range(len(x)):
                    clusters[x[i]].append(i)
                self.birch[number_of_clusters] = [clusters, clustering]
            else:
                clusters, clustering = self.birch[number_of_clusters]
            label = clustering.predict([transformed_instance])[0]
            neighbours = []
            for i in clusters[label]:
                neighbours.append(self.initial_data[i])

            neighbours = np.array(neighbours)
            neighbours = np.concatenate((np.array([instance]),
                                         neighbours))  # when we can't get the reduced dimensions for the instance's embeddings
            return neighbours
        else:
            neighbours_per_dim = {}
            for j in range(0, self.number_of_reduced_dims):
                if tuple([number_of_clusters, j]) not in self.birch:
                    clustering = Birch(
                        n_clusters=number_of_clusters).fit(
                        copy_of_initial[:, j].reshape((len(copy_of_initial[:, j]), 1)))
                    x = clustering.labels_
                    clusters = {}
                    for i in set(x):
                        clusters[i] = []
                    for i in range(len(x)):
                        clusters[x[i]].append(i)
                    self.birch[tuple([number_of_clusters, j])] = [clusters, clustering]
                else:
                    clusters, clustering = self.birch[tuple([number_of_clusters, j])]
                label = clustering.predict(np.array([transformed_instance[j]]).reshape(1, -1))[0]
                neighbours = []
                for i in clusters[label]:
                    neighbours.append(self.initial_data[i])
                neighbours = np.array(neighbours)
                neighbours_per_dim[j] = np.concatenate((np.array([instance]), np.array(neighbours)))
            return neighbours_per_dim

    def explain_instance(self,
                         instance,
                         number_of_neighbours='auto',
                         auto_alpha=True,
                         ng_technique='KNN'):
        """Generates explanations (weights) for an instance.
        
        First, the neighbourhood of each instance is initialised. Then, we learn
        linear models locally using the above neighbourhoods to explain each of
        the instances.

        Args:
            instance: 1d array corresponding to the instance we want to be 
                explained.
            number_of_neighbours: the number of neighours we want to use to 
                achieve the local interpretation. If 'auto' the number will be 
                chosen depending on the dataset.
            auto_alpha: if True, function best_alpha will be used to find most 
                suitable alpha for the linear model. If False, default value of
                alpha will be used (alpha=1.0).
            use_LIME: if True, function give_me_the_neighbourhood of 
                LimeTabularExplainer is used for the neighbourhood extraction.
                If False, sklearn NearestNeighbors() will be used. 

        Returns:
            _coef_per_component: 2d array of shape (n_components, n_features)
                containing weights where n_components is the dimension of the 
                reduced space, and n_features is the number of dimensions of 
                the original space.
        """
        # initialize neighbourhood
        if number_of_neighbours == 'auto':
            number_of_neighbours = self._choose_number_of_neighbours(instance)
        if ng_technique == 'KNN':
            neighbours = self._ng_knn(instance, number_of_neighbours)
        elif ng_technique == 'LatentKNN':
            neighbours = self._ng_latent_knn(instance, number_of_neighbours)
        elif ng_technique == 'Clustering':
            neighbours = self._ng_clustering(instance, number_of_neighbours)
        elif ng_technique == 'LatentClustering':
            neighbours = self._ng_latent_clustering(instance, number_of_neighbours)
        else:  # Global!
            neighbours = self.initial_data

        # transform neighbors
        if self.type != 'locallocal':
            transformed = self._pseudo_transform(neighbours)
        else:
            transformed = {}
            for i in range(0, self.number_of_reduced_dims):
                transformed[i] = self._pseudo_transform(neighbours[i])

        if ng_technique == 'Global':
            _coef_per_component = self._get_coef(neighbours, transformed, None, False)
        else:
            if self.type != 'locallocal':
                distances = euclidean_distances([neighbours[0]], neighbours)[0]
                _coef_per_component = self._get_coef(neighbours, transformed, distances, auto_alpha)
            else:
                distances = {}
                for i in range(0, self.number_of_reduced_dims):
                    distances[i] = euclidean_distances([neighbours[i][0]], neighbours[i])[0]
                _coef_per_component = self._get_coef(neighbours, transformed, distances, auto_alpha)
        return _coef_per_component

    def explain_reduced_instance(self,
                                 reduced_instance,
                                 number_of_neighbours='auto',
                                 auto_alpha=True,
                                 ng_technique='KNN'):
        """Generates explanations (weights) for an instance.
        
        First, the neighbourhood of each instance is initialised. Then, we learn
        linear models locally using the above neighbourhoods to explain each of
        the instances.

        Args:
            instance: 1d array corresponding to the instance we want to be 
                explained.
            number_of_neighbours: the number of neighours we want to use to 
                achieve the local interpretation. If 'auto' the number will be 
                chosen depending on the dataset.
            auto_alpha: if True, function best_alpha will be used to find most 
                suitable alpha for the linear model. If False, default value of
                alpha will be used (alpha=1.0).
            use_LIME: if True, function give_me_the_neighbourhood of 
                LimeTabularExplainer is used for the neighbourhood extraction.
                If False, sklearn NearestNeighbors() will be used. 

        Returns:
            _coef_per_component: 2d array of shape (n_components, n_features)
                containing weights where n_components is the dimension of the 
                reduced space, and n_features is the number of dimensions of 
                the original space.
        """
        if ng_technique == 'LatentKNN':
            neighbours = self._ng_latent_knn_inverse(reduced_instance, number_of_neighbours)
        else:  # Global!
            print('Not implemented yet. Only LatentKNN so far')

        # transform neighbors
        if self.type != 'locallocal':
            transformed = self._pseudo_transform(neighbours)
        else:
            transformed = {}
            for i in range(0, self.number_of_reduced_dims):
                transformed[i] = self._pseudo_transform(neighbours[i])

        if ng_technique == 'Global':
            _coef_per_component = self._get_coef(neighbours, transformed, None, False)
        else:
            if self.type != 'locallocal':
                distances = euclidean_distances([reduced_instance], transformed)[0]
                _coef_per_component = self._get_coef(neighbours, transformed, distances, auto_alpha)
            else:
                distances = {}
                for i in range(0, self.number_of_reduced_dims):
                    distances[i] = euclidean_distances([reduced_instance], transformed[i])[0]
                _coef_per_component = self._get_coef(neighbours, transformed, distances, auto_alpha)
        return _coef_per_component

    def visualise_weights(self,
                          instance,
                          number_of_neighbours='auto',
                          auto_alpha=True,
                          use_LIME=False,
                          dimension=0):
        """Visual comparison of weights.
        
        This function only works when the dimensionality reduction technique is 
        able to return weights per each component. Creates a plot that 
        compares weights extracted from the linear model and the dimensionality
        reduction technique, of a particular dimension of the instance.

        Args:
            instance: 1d array representing a particular sample
            number_of_neighbours: the number of neighours we want to use to 
                achieve the local interpretation. If 'auto' the number will be 
                chosen depending on the dataset.
            auto_alpha: if True, function best_alpha will be used to find most 
                suitable alpha for the linear model. If False, default value of
                alpha will be used (alpha=1.0).
            use_LIME: if True, function give_me_the_neighbourhood of 
                LimeTabularExplainer is used for the neighbourhood extraction.
                If False, sklearn NearestNeighbors() will be used. 
            dimension: the dimension we wont to visualise
            normalised: if True, data will be divided with the absolute max 
                value of the data.
        """
        if self.model.components_ is not None:
            original_coef_ = self.model.components_[dimension]
        else:
            print(
                'This DR technique does not provide weights per component. It is a black box. We cannot compute the weights error. Try reconstruction error instead! Cheers..'
            )
            return None

        if number_of_neighbours == 'auto':
            number_of_neighbours = self._choose_number_of_neighbours(
                instance)

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                          auto_alpha, use_LIME)

        plotdata = pd.DataFrame({
            "Model": dr_coefs_[dimension],
            "PCA": original_coef_
        },
            index=self.feature_names)
        plt.rcParams["figure.figsize"] = [7, 7]
        plt.rcParams["figure.dpi"] = 200
        plotdata.plot(kind="bar", rot=0)
        plt.title("Weights comparison")
        plt.xlabel("Features")
        plt.ylabel("Weights")
        plt.show()

    def visualise_reconstructed(self,
                                instance,
                                number_of_neighbours='auto',
                                auto_alpha=True,
                                use_LIME=False):
        """Visual comparison of dimensionally reduced instances.
        
        Reconstructs the dimensionally redueced data throught the weights extracted
        from the linear model, and compares them with the dimensionally reduced 
        given from the dimensionality reduction technique for a particular 
        instance.

        Args:
            instance: 1d array representing a particular sample
            number_of_neighbours: the number of neighours we want to use to 
                achieve the local interpretation. If 'auto' the number will be 
                chosen depending on the dataset.
            auto_alpha: if True, function best_alpha will be used to find most 
                suitable alpha for the linear model. If False, default value of
                alpha will be used (alpha=1.0).
            use_LIME: if True, function give_me_the_neighbourhood of 
                LimeTabularExplainer is used for the neighbourhood extraction.
                If False, sklearn NearestNeighbors() will be used. 
            use_mean: if True, mean value of the data will be subtracted for 
                each instance.
            normalised: if True, data will be divided with the absolute max 
                value of the data.
        """
        if number_of_neighbours == 'auto':
            number_of_neighbours = self._choose_number_of_neighbours(
                instance)

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                          auto_alpha, use_LIME)

        transformed = self._pseudo_transform(np.array([instance]))[0]

        mean = self.initial_data.mean(axis=0)
        instance = instance - self.mean

        lxdr_tr = []
        for coef in dr_coefs_:
            lxdr_tr.append(sum(instance * coef))
        lxdr_tr = np.array(lxdr_tr)

        components = []
        for i in range(1, len(dr_coefs_) + 1):
            components.append("C" + str(i))

        plotdata = pd.DataFrame({"LXDR": lxdr_tr, "PCA": transformed})
        plt.rcParams["figure.figsize"] = [6, 5]
        plt.rcParams["figure.dpi"] = 200
        plotdata.plot(kind="bar", rot=0)
        plt.title("Values comparison")
        plt.xlabel("Components")
        plt.ylabel("Values")
        # plt.axes.set_xticklabels(components)
        plt.show()

    def create_heatmap(self,
                       instance,
                       number_of_neighbours='auto',
                       auto_alpha=True,
                       use_LIME=False):
        """Heatmap weight comparison.
        
        Plots a Heatmap showing the difference between weigths extracted through
        the linear model and weigths extracted from the dimensionality reduction
        technique, for a particular instance.

        Args:
            instance: 1d array representing a particular sample
            number_of_neighbours: the number of neighours we want to use to 
                achieve the local interpretation. If 'auto' the number will be 
                chosen depending on the dataset.
            auto_alpha: if True, function best_alpha will be used to find most 
                suitable alpha for the linear model. If False, default value of
                alpha will be used (alpha=1.0).
            use_LIME: if True, function give_me_the_neighbourhood of 
                LimeTabularExplainer is used for the neighbourhood extraction.
                If False, sklearn NearestNeighbors() will be used. 
            normalised: if True, data will be divided with the absolute max 
                value of the data.
        """
        if self.model.components_ is not None:
            original_coefs_ = self.model.components_
        else:
            print(
                'This DR technique does not provide weights per component. It is a black box. We cannot compute the weights error. Try reconstruction error instead! Cheers..'
            )
            return None

        if number_of_neighbours == 'auto':
            number_of_neighbours = self._choose_number_of_neighbours(
                instance)

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                          auto_alpha, use_LIME)

        # this is used for labeling
        components = []
        for i in range(1, len(dr_coefs_) + 1):
            components.append("C" + str(i))

        scores = abs(np.subtract(original_coefs_, dr_coefs_)) * 1000
        plt.rcParams["figure.figsize"] = [6, 4.5]
        plt.rcParams["figure.dpi"] = 200
        ax = sns.heatmap(scores,
                         annot=True,
                         fmt='.2f',
                         linewidth=0.1,
                         cmap='hot_r',
                         xticklabels=self.feature_names,
                         yticklabels=components)

        plt.xlabel('Features')
        plt.ylabel('Components')
        plt.title("Error between weights")
        plt.show()


def ig(sample, model, reduced_dimensions):
    baseline = tf.zeros(shape=(len(sample)))
    m_steps = 100
    alphas = tf.linspace(start=-1.0, stop=1.0, num=m_steps + 1)

    def interpolate_samples(baseline, sample, alphas):
        alphas_x = alphas[:, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        sample = tf.convert_to_tensor(sample, dtype='float')
        input_x = tf.expand_dims(sample, axis=0)
        delta = input_x - baseline_x
        samples = baseline_x + alphas_x * delta
        return samples

    isamples = interpolate_samples(
        baseline=baseline, sample=sample, alphas=alphas)

    def compute_gradients(samples, target_label):
        with tf.GradientTape() as tape:
            tape.watch(samples)
            logits = model(samples)[:, target_label]
        return tape.gradient(logits, samples)

    def integral_approximation(gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    weights = []
    for dim in range(reduced_dimensions):
        path_gradients = compute_gradients(samples=isamples, target_label=dim)
        igd = integral_approximation(gradients=path_gradients)
        weights.append(igd.numpy())
    return weights
