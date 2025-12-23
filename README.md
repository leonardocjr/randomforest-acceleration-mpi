# Random Forests Acceleration with MPI
Goal: To accelerate a sequential implementation of the Random Forests method using MPI

Random Forests build an "ensemble" (a group) of many individual decision trees to make more accurate predictions for both classification (e.g., spam/not spam) and regression (e.g., predicting house prices) tasks, by using random subsets of data and features to train each tree, then 
combining their results through voting or averaging for a robust final answer. They reduce the variance and overfitting issues of single decision trees, making them stable and reliable algorithms. How They Work: 

1) Bootstrap Aggregation (Bagging): Each tree in the forest is trained on a random 
subset of the original training data (sampled with replacement);
2) Feature Randomness: At each split point in a tree, only a random subset of features (variables) is considered, not all of them, which decorrelates the trees;
3) Ensemble Voting/Averaging: in Classification, the final prediction is determined by majority vote (the class predicted by most trees); in Regression, the final prediction is the average of the predictions from all the individual trees.

**Advisor:** <a href="https://www.cienciavitae.pt/portal/C414-F47F-6323">José Carlos Rufino Amaro</a><br/>
**Instituição de Ensino:** [Instituto Politécnico de Bragança](IPB.pt)
