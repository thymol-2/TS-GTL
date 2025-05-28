# TS-GTL
Task Similarity-guided Transfer Learning based on Molecular Graphs


# Abstract
Drug absorption significantly influences pharmacokinetics. Accurately predicting human oral bioavailability (HOB) is essential for optimizing drug candidates and improving clinical success rates. The traditional method based on experiment is a common way to obtain HOB, but the experimental method is time-consuming and costly. Recently, using AI models to predict ADMET properties has become a new and effective method. However, this method has some data dependence problems. To address this issue, we combine physicochemical properties with graph-based deep learning methods to improve HOB prediction, providing an efficient and interpretable alternative to traditional experimental and computational approaches for ADMET property studies in data-scarce scenarios. We propose a similarity-guided transfer learning framework, Task Similarity-guided Transfer Learning based on Molecular Graphs (TS-GTL), which includes a deep learning model, PGnT (pKa Graph-based Knowledge-driven Transformer). PGnT incorporates common molecular descriptors as external knowledge to guide molecular graph representation, leveraging GNNs and Transformer encoders to enhance feature extraction. Additionally, we introduce MoTSE to quantify the similarity between physicochemical properties and HOB. Notably, training with data pre-trained model on logD properties showed the best performance in transfer learning. TS-GTL also outperformed machine learning algorithms and deep learning predictive tools, underscoring the critical role of task similarity in transfer learning. 


# Usage method
Running File 1 can train the model
Running File 2 can achieve transfer learning
Running File 3 can calculate the similarity between tasks
