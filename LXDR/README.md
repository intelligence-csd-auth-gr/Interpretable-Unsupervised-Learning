# LXDR
Exploring Local Interpretability in Dimensionality Reduction: Analysis and Use Cases

Dimensionality reduction is a crucial area in artificial intelligence that enables the visualization and analysis of high-dimensional data. A use of dimensionality reduction is to lower the dimensional complexity of data, improving the performance of machine learning models. Non-linear dimensionality reduction approaches, which provide higher quality representations than linear ones, lack interpretability, prohibiting their application in tasks requiring interpretability. This paper presents LXDR (Local eXplanation of Dimensionality Reduction), a local, model-agnostic technique that can be applied to any DR technique. LXDR trains linear models around a neighborhood of a specific instance and provides local interpretations using a variety of neighborhood generation techniques. Variations of the proposed technique are also introduced. The effectiveness of LXDR's interpretations is evaluated by quantitative and qualitative experiments, as well as demonstrations of its practical implementation in diverse use cases. The experiments emphasize the importance of interpretability in dimensionality reduction and how LXDR reinforces it.

## Requirements
- For the quantitative analysis experiments, and for the two use cases (1 - supervised regression and 2 - topic representationn), we used req.txt, and we run the experiments in a docker build with the instructions apparent on "Dockerfile".
- For the Extreme multi-label classification use case (3rd), we use the req.txt python libraries, and run the experiments in Colab Pro.

## Use cases
1. Supervised Regression: The objective of this use case is twofold. Firstly, we aim to demonstrate how LXDR can be utilized to obtain feature importance explanations for the predictions made by a regression model trained on dimensionally reduced data. Additionally, we aim to evaluate the accuracy and faithfulness of these explanations.
2. Topic Representation: Topic representation enables us to generate embeddings for lengthy documents. This is done by identifying key topics for each document, which are then used to create the final representation. Topic embeddings can be utilized for both unsupervised and supervised tasks. In this example, we will demonstrate how LXDR can be used in a multi-class classification problem with topic embeddings.
3. Extreme Multi-Label Classification: In this use case, we are showcasing how LXDR can be applied to an extreme multi-label classification task. We will use the [BioASQ challenge datasets](http://bioasq.org/participate/challenges), which consist of biomedical publications indexed with Medical Subject Headings (MeSH), and are a set of data that fits within the extreme multi-label classification paradigm. Each dataset contains over a million documents and focuses on a particular year of the MeSH vocabulary, ranging from 2013 to 2023 (the most recent). The articles are labeled with MeSH descriptors, with each one having an average of 13 descriptors among a pool of over 30,000. We examined a subset of the 2020 BioASQ dataset for this use case, which contained approximately 45k articles indexed with 17k descriptors, with each article being associated with 14 of them on average.

## Contributors on LXDR
Name | Email
--- | ---
| Nikolaos Mylonas | myloniko@csd.auth.gr |
| Ioannis Mollas | iamollas@csd.auth.gr |
| Grigorios Tsoumakas | greg@csd.auth.gr |
| Nick Bassiliades | nbassili@csd.auth.gr |

## Cite our Work
Paper submitted to IEEE Access. Until it is published, please cite the followinf paper:: [Local Explanation of Dimensionality Reduction](https://arxiv.org/abs/2204.14012)

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
