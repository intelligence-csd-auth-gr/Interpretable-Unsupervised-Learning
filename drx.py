"""
Functions for dimensionality reduction explanation.
"""
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from lime.lime_tabular import LimeTabularExplainer


class DRX:

    def __init__(self,
                 dimensionality_reduction_model,
                 feature_names,
                 initial_data=None):
        """Init function.

        Args:
            dimensionality_reduction_model: model used to dimensionally reduce 
                the data (should be fitted on the initial dataset prior the 
                object creation).
            feature_names: The names of the dataset columns.
            initial_data: 2d array
        """
        self.model = dimensionality_reduction_model
        self.feature_names = feature_names
        self.lime = LimeTabularExplainer(training_data=initial_data,
                                         discretize_continuous=False,
                                         mode="regression",
                                         random_state=0)
        self.knn = NearestNeighbors()
        self.knn.fit(initial_data)
        self.initial_data = initial_data

    def normalise_(self, array):
        array = np.array(array)
        max_value = np.max(array)
        min_value = np.min(array)
        abs_max = max(abs(max_value), abs(min_value))
        return array / abs_max

    def kernel(self, array, d):  #IM: Changed this function
        normilised_distances = []
        for sample in array:
            normilised_distances.append(
                math.exp(-(sample * math.log(d) / 2)) * math.log(d))
        return np.array(normilised_distances)

    def best_alpha(self, X, y, distances):  #IM: Added here the distances
        best_model = None
        best_score = sys.maxsize
        alphas = [0.1, 1.0, 10]
        best_a = 0
        performances = []
        for a in alphas:
            temp_model = Ridge(alpha=a).fit(X, y, distances)
            temp_perfomance = abs(mean_squared_error(y, temp_model.predict(X)))
            performances.append(temp_perfomance)
            if temp_perfomance < best_score:
                best_score = temp_perfomance
                best_model = temp_model
                best_a = a
        return best_model  #, best_a, performances

    def _choose_number_of_neighbours(self, instance):
        return int(len(self.initial_data)/10)

    def _get_coef(self, neighbours, transformed, distances, auto_alpha):
        reduced_dimensions = len(transformed[0])
        _coef_per_component = []
        for component in range(reduced_dimensions):
            if auto_alpha:
                linear_model = self.best_alpha(neighbours,
                                               transformed[:, component],
                                               distances)
            else:
                linear_model = Ridge(alpha=1.0).fit(
                    neighbours, transformed[:, component],
                    distances)  #, normilised_distances[sample])
            _coef_per_component.append(linear_model.coef_)
        return np.array(_coef_per_component)

    def explain_instance(self,
                         instance,
                         number_of_neghbours='auto',
                         auto_alpha=True,
                         use_LIME=False):
        """Generates explanations for an instance.
        
        First, the neighbourhood of each instance is initialised. Then, we learn
        linear models locally using the above neighbourhoods to explain each of
        the instances.

        Args:
            instance: 1d array corresponding to the instance we want to be 
                explained.
            number_of_neghbours: the number of neighours we want to use to 
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
        if number_of_neghbours == 'auto':
            number_of_neghbours = self._choose_number_of_neighbours(
                instance)  #Something smarter here!
        if use_LIME:
            neighbours = self.lime.give_me_the_neighbourhood(
                instance,
                self.model.transform,
                num_samples=number_of_neghbours)[0]

        else:
            neighbours = []
            neighbours_ind = self.knn.kneighbors([instance],
                                                 number_of_neghbours,
                                                 return_distance=False)[0]
            for i in neighbours_ind:
                neighbours.append(self.initial_data[i])
            neighbours = np.array(neighbours)

        neighbours = np.concatenate((np.array([instance]), neighbours))

        transformed = self.model.transform(neighbours)

        distances = euclidean_distances([transformed[0]], transformed)[0]

        distances = self.kernel(distances, len(instance))

        _coef_per_component = self._get_coef(neighbours, transformed,
                                             distances, auto_alpha)

        return _coef_per_component

    def find_reconstruction_error(self,
                                  data,
                                  number_of_neighbours='auto',
                                  auto_alpha=True,
                                  use_LIME=False,
                                  use_mean=True,
                                  normalised=False):
        """Prints reconstruction error  

        Prints mean absolute error, cosine error, euclidean error between the 
        reduced data extracted through the linear model and the dimensionality 
        reduction technique.

        Args:
            data: 1d or 2d array. If 1d, shape (-1, 1) should be met.
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
            
        Returns:
            A tuple (mae, cos_error, euc_error), where:
                mae: mean absolute error (use of sklearn mean_absolute_error)
                cos_error: cosine error (use of sklearn cosine_similarity)
                euc_error: euclidean error (use of sklearn euclidean_distances)
        """
        if number_of_neighbours == 'auto':
            number_of_neghbours = self._choose_number_of_neighbours(
                data[0])  #Something smarter here!

        transformed = self.model.transform(data)
        drx_transformed = []
        mean = self.initial_data.mean(axis=0)
        for d in data:
            coefs_ = self.explain_instance(d, number_of_neighbours, auto_alpha,
                                           use_LIME)
            d_new = d
            if use_mean:
                d_new = d_new - mean

            drx_tr = []
            for coef in coefs_:
                drx_tr.append(sum(d_new * coef))
            drx_transformed.append(np.array(drx_tr))

        mae = mean_absolute_error(transformed, drx_transformed)
        print("mean_absolute_error = ", mae)

        cos_sum = 0
        euc_sum = 0
        for sample in range(len(transformed)):
            ts = transformed[sample]
            dr = drx_transformed[sample]
            if normalised:
                ts = self.normalise_(ts)
                dr = self.normalise_(dr)
            cos_sum += 1 - cosine_similarity(np.array([ts]), np.array([dr]))
            euc_sum += euclidean_distances(np.array([ts]), np.array([dr]))
        cos_error = cos_sum / len(transformed)
        euc_error = euc_sum / len(transformed)

        print("cosine error = ", cos_error[0][0])
        print("euclidean error = ", euc_error[0][0])

        return mae, cos_error, euc_error

    def find_weights_error(self,
                           data,
                           number_of_neighbours='auto',
                           auto_alpha=True,
                           use_LIME=False,
                           normalised=False):
        """Prints weight error  

        This function only works when the dimensionality reduction technique is 
        able to return weights per each component. Prints mean absolute 
        error, cosine error, euclidean error between the weigths created through
        the linear model, and weights extracted through the dimensionality 
        reduction technique.

        Args:
            data: 2d array
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
            
        Returns:
            A tuple (mae, cos, euc), where:
                mae: mean absolute error (use of sklearn mean_absolute_error)
                cos: cosine error (use of sklearn cosine_similarity)
                euc: euclidean error (use of sklearn euclidean_distances)
        """
        if self.model.components_ is not None:
            original_coefs_ = self.model.components_
        else:
            print(
                'This DR technique does not provide weights per component. It is a black box. We cannot compute the weights error. Try reconstruction error instead! Cheers..'
            )
            return None

        if number_of_neighbours == 'auto':
            number_of_neghbours = self._choose_number_of_neighbours(
                data[0])  #Something smarter here!

        transformed = self.model.transform(data)
        mae = 0
        cos = 0
        euc = 0
        count = 0
        for d in data:
            coefs_ = self.explain_instance(d, number_of_neighbours, auto_alpha,
                                           use_LIME)
            for dimension in range(len(coefs_)):
                oc = original_coefs_[dimension]
                c = coefs_[dimension]
                if normalised:
                    oc = self.normalise_(oc)
                    c = self.normalise_(c)
                mae += mean_absolute_error(np.array([oc]), np.array([c]))
                cos += 1 - cosine_similarity(np.array([oc]), np.array(
                    [c]))[0][0]
                euc += euclidean_distances(np.array([oc]), np.array([c]))[0][0]
                count += 1

        mae = mae / count
        cos = cos / count
        euc = euc / count

        print("mean_absolute_error = ", mae)
        print("cosine error = ", cos)
        print("euclidean error = ", euc)

        return mae, cos, euc

    def visualise_weights(self,
                          instance,
                          number_of_neighbours='auto',
                          auto_alpha=True,
                          use_LIME=False,
                          dimension=0,
                          normalised=False):
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
                instance)  #Something smarter here!

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                          auto_alpha, use_LIME)

        if normalised:
            original_coef_ = self.normalise_(original_coef_)
            dr_coef_ = self.normalise_(dr_coefs_[dimension])

        plotdata = pd.DataFrame({
            "Model": dr_coef_,
            "PCA": original_coef_
        },
                                index=self.feature_names)
        plt.rcParams["figure.figsize"] = [6, 6]
        plotdata.plot(kind="bar", rot=0)
        plt.title("value comparison")
        plt.xlabel("features")
        plt.ylabel("values")
        plt.show()

    def visualise_reconstructed(self,
                                instance,
                                number_of_neighbours='auto',
                                auto_alpha=True,
                                use_LIME=False,
                                use_mean=True,
                                normalised=True):
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
                instance)  #Something smarter here!

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                          auto_alpha, use_LIME)

        transformed = self.model.transform(np.array([instance]))[0]

        mean = self.initial_data.mean(axis=0)

        d_new = instance
        if use_mean:
            d_new = d_new - mean

        drx_tr = []
        for coef in dr_coefs_:
            drx_tr.append(sum(d_new * coef))
        drx_tr = np.array(drx_tr)

        if normalised:
            drx_tr = self.normalise_(drx_tr)
            transformed = self.normalise_(transformed)

        plotdata = pd.DataFrame({"Model": drx_tr, "PCA": transformed})
        plt.rcParams["figure.figsize"] = [6, 6]
        plotdata.plot(kind="bar", rot=0)
        plt.title("value comparison")
        plt.xlabel("latent features")
        plt.ylabel("values")
        plt.show()

    def create_heatmap(self,
                       instance,
                       number_of_neighbours='auto',
                       auto_alpha=True,
                       use_LIME=False,
                       normalised=False):
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
                instance)  #Something smarter here!

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                          auto_alpha, use_LIME)

        if normalised:
            original_coefs_ = np.array([
                self.normalise_(original_coef_)
                for original_coef_ in original_coefs_
            ])
            dr_coefs_ = np.array(
                [self.normalise_(dr_coef_) for dr_coef_ in dr_coefs_])

        components = []
        for i in range(len(dr_coefs_)):
            components.append("C" + str(i))

        scores = abs(np.subtract(original_coefs_, dr_coefs_))
        ax = sns.heatmap(scores,
                         annot=True,
                         fmt='.2f',
                         linewidth=0.1,
                         cmap='hot_r',
                         xticklabels=self.feature_names,
                         yticklabels=components)

        plt.xlabel('Features')
        plt.ylabel('Components')
        plt.title("error between weights")
        plt.show()
