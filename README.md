#### This is a repository for CTR prediction
- models include: 
  - adaboost (poor performance)
  - decision tree (low F1 score)
  - MLP (balance F1 and accuracy)
  - DeepFM (under construction)
- data visualization using t-SNE
- data augmentation using SMOTE (Synthesis Minority Over-Sampling Technique)
#### File in model directory
- `Decision_Tree.ipynb`: using decision tree classifier
- `adaboost.ipynb`: using adaboost classifier
- `DeepFM.py`: using DeepFM model in deepctr python module
- `Final_MLP.ipynb`: using final MLP for classification
**Notice: running DeepFM.py would require the installation of `deepctr`**
