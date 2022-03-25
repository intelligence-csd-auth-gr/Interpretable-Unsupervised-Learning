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
from sklearn.metrics import mean_squared_error, mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras import layers, losses, callbacks, Sequential
from tensorflow.keras.models import Model
import keras.backend as K

class LXDR:
    """Local Explanation of Dimensionality Reduction (LXDR)

    A technique for interpreting locally the results of DR techniques.
    
    """
    
    def __init__(self,
                 dimensionality_reduction_model,
                 feature_names,
                 scope = 'local',
                 initial_data=None,
                 neural=False,
                 mean=None):
        """Init function.

        Args:
        dimensionality_reduction_model: model used to dimensionally reduce 
                    the data (should be fitted on the initial dataset prior the 
                    object creation).
        feature_names: The names of the dataset column
        scope: If the techinque will work locally or globally. Accepted values 'local', or 'global'.
        initial_data: 2d array contaning the orgina dataset
        neural: if the DR is an AutoEncoder
        mean: if mean is going to be used during the training of the local models
        """
        
        self.model = dimensionality_reduction_model
        self.feature_names = feature_names
        self.scope = scope
        if scope == 'local':
            self.lime = LimeTabularExplainer(training_data=initial_data,
                                         discretize_continuous=False,
                                         mode="regression",
                                         random_state=0)
            self.knn = NearestNeighbors()
            self.knn.fit(initial_data)
        self.initial_data = initial_data
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.zeros(initial_data[0].shape)
        self.neural = neural
        self.global_ = {}

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
        alphas = [0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5, 1.9, 10, 100, 1000]
        best_a = 0
        performances = []
        for a in alphas:
            temp_model = Ridge(alpha=a, fit_intercept=False, random_state=7).fit(X, y)
            temp_perfomance = mae(y, temp_model.predict(X), sample_weight=distances)
            performances.append(temp_perfomance)
            if temp_perfomance < best_score:
                best_score = temp_perfomance
                best_model = temp_model
                best_a = a
        return best_model

    def _choose_number_of_neighbours(self, instance):
        """ Simpel funtion determining the number of neighbors in case it is not defined 
          by the user.

          Returns:
            int: number of neighbors
        """
        return int(3*len(self.initial_data)/4)

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
        reduced_dimensions = len(transformed[0])
        _coef_per_component = []
                
        for component in range(reduced_dimensions):
            tc = transformed[:, component]
            if auto_alpha:
                linear_model = self.best_alpha(neighbours-self.mean,tc,distances)
            else:
                if distances is not None:
                    linear_model = Ridge(alpha=1.0, fit_intercept=False, random_state=7).fit(neighbours-self.mean, tc, distances)
                else: #is Global!
                    if component not in self.global_:
                        linear_model = Ridge(alpha=1.0, fit_intercept=False, random_state=7).fit(neighbours-self.mean, tc)
                        self.global_[component] = linear_model
                    else:
                        linear_model = self.global_[component]
            _coef_per_component.append(linear_model.coef_)
        return np.array(_coef_per_component)

    def explain_instance(self,
                         instance,
                         number_of_neighbours='auto',
                         auto_alpha=True,
                         use_LIME=False):
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
        if self.scope == 'local':
            if number_of_neighbours == 'auto':
                number_of_neighbours = self._choose_number_of_neighbours(
                    instance)
            if use_LIME:
                if self.neural:
                    neighbours = self.lime.give_me_the_neighbourhood(
                    instance,
                    self.model.predict,
                    num_samples=number_of_neighbours)[0]
                else:
                    neighbours = self.lime.give_me_the_neighbourhood(
                    instance,
                    self.model.transform,
                    num_samples=number_of_neighbours)[0]

            else:# NG=='KNNs'
                neighbours = []
                neighbours_ind = self.knn.kneighbors([instance],
                                                     number_of_neighbours,
                                                     return_distance=False)[0]
                for i in neighbours_ind:
                    neighbours.append(self.initial_data[i])
                neighbours = np.array(neighbours)
            neighbours = np.concatenate((np.array([instance]), neighbours))
        else:
            neighbours = self.initial_data
        
        # transform neighbors
        if self.neural:
            transformed = self.model.predict(neighbours)
        else:
            transformed = self.model.transform(neighbours)
        
        # create weights
        if self.scope == 'local':
            distances = euclidean_distances([neighbours[0]], neighbours)[0]
            normilised_distances = []
            for sample in distances:
                normilised_distances.append(
                    math.exp(-(2*sample)))
            distances= np.array(normilised_distances)
            
            _coef_per_component = self._get_coef(neighbours, transformed, distances, auto_alpha)
        else:
            _coef_per_component = self._get_coef(neighbours, transformed, None, False)
        return _coef_per_component

    def find_reconstruction_error(self,
                                  data,
                                  number_of_neighbours='auto',
                                  auto_alpha=True,
                                  use_LIME=False):
        """Prints reconstruction error  

        Prints mean absolute error, cosine error, euclidean error between the 
        reduced data extracted through the linear model and the dimensionality 
        reduction technique.

        Args:
            data: 1d or 2d array. If 1d, shape (1, -1) should be met.
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
        euc_error: euclidean error (use of sklearn euclidean_distances)
        """
        if number_of_neighbours == 'auto':
            number_of_neighbours = self._choose_number_of_neighbours(
                data[0])

        transformed = self.model.transform(data)
        lxdr_transformed = []
        mean = self.initial_data.mean(axis=0)
        for d in data:
            coefs_ = self.explain_instance(d, number_of_neighbours, auto_alpha,
                                           use_LIME)
            # mean center the instances
            d_new = d
            d_new = d_new - self.mean

            # By multiplying the mean centred insances with the coef_,
            # we are able to create the reduced space data (lxdr_transformed).
            lxdr_tr = []
            for coef in coefs_:
                lxdr_tr.append(sum(d_new * coef))
            lxdr_transformed.append(np.array(lxdr_tr))

        cos_sum = 0
        euc_sum = 0
        for sample in range(len(transformed)):
            ts = transformed[sample]
            dr = lxdr_transformed[sample]
            cos_sum += 1 - cosine_similarity(np.array([ts]), np.array([dr]))
            euc_sum += euclidean_distances(np.array([ts]), np.array([dr]))
        
        # calculate mean_absolute_error, cosine error and euclidean error
        cos_error = cos_sum / len(transformed)
        euc_error = euc_sum / len(transformed)
        mae = mean_absolute_error(transformed, lxdr_transformed)

        print("mean_absolute_error = ", mae)
        print("cosine error = ", cos_error[0][0])
        print("euclidean error = ", euc_error[0][0])

        return mae, cos_error, euc_error

    def find_weights_error(self,
                           data,
                           number_of_neighbours='auto',
                           auto_alpha=True,
                           use_LIME=False):
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
            number_of_neighbours = self._choose_number_of_neighbours(
                data[0])

        transformed = self.model.transform(data)
        mae = 0
        cos = 0
        euc = 0
        count = 0
        for d in data:
            coefs_ = self.explain_instance(d, number_of_neighbours, auto_alpha,
                                           use_LIME)
            
            # caclulate error between weights provided by the DR technique (original_coefs_)
            # and weights provided by lxdr (coefs_)
            for dimension in range(len(coefs_)):
                oc = original_coefs_[dimension]
                c = coefs_[dimension]
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

        transformed = self.model.transform(np.array([instance]))[0]

        mean = self.initial_data.mean(axis=0)
        instance = instance - self.mean

        lxdr_tr = []
        for coef in dr_coefs_:
            lxdr_tr.append(sum(instance * coef))
        lxdr_tr = np.array(lxdr_tr)

        components = []
        for i in range(1,len(dr_coefs_)+1):
            components.append("C" + str(i))
            
        plotdata = pd.DataFrame({"LXDR": lxdr_tr, "PCA": transformed})
        plt.rcParams["figure.figsize"] = [6, 5]
        plt.rcParams["figure.dpi"] = 200
        plotdata.plot(kind="bar", rot=0)
        plt.title("Values comparison")
        plt.xlabel("Components")
        plt.ylabel("Values")
        #plt.axes.set_xticklabels(components)
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
        for i in range(1,len(dr_coefs_)+1):
            components.append("C" + str(i))

        scores = abs(np.subtract(original_coefs_, dr_coefs_))*1000
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

    def get_feature_importance(self, 
                               instance, 
                               number_of_neighbours='auto', 
                               auto_alpha=False, 
                               use_LIME=False):
        """ Creates and printd an array shaped (N_features, N_reduced_features) 
        where for each reduced features, highlights the original feature that 
        contributed the most to its creation.
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
        """

        dr_coefs_ = self.explain_instance(instance, number_of_neighbours,
                                            auto_alpha, use_LIME)

        # mean centre the instance
        instance = instance - self.mean

        # get the feature contribution 
        all_values = []
        for i in range(len(dr_coefs_)):
          value_with_drx = instance  *  dr_coefs_.T[:,i]
          all_values.append(value_with_drx)

        # display a matrix with the feature imporance (how much has each feature
        # contributed for the creation of each reduced representation)
        # The feature with the greater contribution is highlighted with green
        # NOTE: for visualization purposes, the values have been converted to their
        #       abs value
        df = pd.DataFrame(np.array([abs(ele) for ele in all_values]).T)
        df = df.style.highlight_max(color = 'green', axis = 0)
        display(df)
