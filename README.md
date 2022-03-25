# LXDR
LXDR: Local Explanation of Dimensionality Reduction

Dimensionality reduction (DR) is a popular method for preparing and analyzing high-dimensional data. Reduced data representations are less computationally intensive and easier to manage and visualize, while retaining a significant percentage of their original information. Aside from these advantages, these reduced representations can be difficult or impossible to interpret in most circumstances, especially when the DR approach does not provide further information about which features of the original space led to their construction. This problem is addressed by Interpretable Machine Learning, a subfield of Explainable Artificial Intelligence that addresses the opacity of machine learning models. However, current research on Interpretable Machine Learning has been focused on supervised tasks, leaving unsupervised tasks like Dimensionality Reduction unexplored. In this paper, we introduce LXDR, a technique capable of providing local interpretations of the output of DR techniques. Experiment results and a LXDR use case example is presented to evaluate its usefulness.

## Instructions
Please, if you want to try LXDR with LIME you will need the follow this procedure:
```
!pip install lime==0.2.0.1
```
Then, replace the file lime_tabular.py to the correspoding directory (e.g. /usr/local/lib/python3.7/dist-packages/lime/)

## Example
```
X, y, feature_names = load_your_data()
dr = DR(n_components=8) #initialize the DR technique you want, and the number of dimensions to reduce to
dr.fit(X)
X_t = dr.transform(X)
mean = X.mean(axis=0)

lxdr = LXDR(dr, feature_names, 'local', X, False, mean)

lxdr.explain_instance(X[1], number_of_neighbours=50, auto_alpha=True, use_LIME=False)
```

## Contributors on LXDR
Name | Email
--- | ---
[X](linkX) | X@email.comm
[X](linkX) | X@email.comm
[X](linkX) | X@email.comm
[X](linkX) | X@email.comm

## Cite our Work
[PAPER](link): Title] Coming

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
