# Dimensionality Reduction eXplained (DRx)

DRx is a technique for interpreting locally the results of Dimensionality Reduction (DR) techniques. DRx extracts interpretations by combining the neighbourhood of an inspected instance with a linear model. It is applicable to any DR technique capable of decreasing new instances without having to be trained on the complete dataset every time.





## Instructions
Please ensure LIME is installed, and lime_tabular.py is moved into LIME's directory. You can find lime_tabular.py [here](https://github.com/intelligence-csd-auth-gr/Interpretable-Predictive-Maintenance/blob/master/TEDS_RUL/lime_tabular.py).
We also provide a code example of DRx. In order to run it, DRx.py and [lime_tabular.py](https://github.com/intelligence-csd-auth-gr/Interpretable-Predictive-Maintenance/blob/master/TEDS_RUL/lime_tabular.py). shall be included in the files.

## Requirements
- ipython==3.7.11
- matplotlib==3.2.2
- seaborn==0.11.2
- pandas==1.3.5
- lime==0.2.0.1
- numpy==1.21.5
- scikit-learn==1.0.2.


## Contributors on VisioRed
| Plugin | README |
| ------ | ------ |
| Avraam Bardos | ampardos@csd.auth.gr |
| Ioannis Mollas | iamollas@csd.auth.gr |
| Grigorios Tsoumakas | greg@csd.auth.gr |
| Nick Bassiliades | nbassili@csd.auth.gr |