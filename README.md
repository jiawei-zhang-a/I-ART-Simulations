# Post-prediction-Causal-Inference
In many practical studies, among the outcomes Y1, . . . , YK , one or multiple of them are
incomplete. In randomized experiments, simply ignoring the missing outcomes may lead to
statistical inference not finite population exact anymore. In matched observational studies,
people routinely exclude study subjects with missing outcomes, which would substantially
the statistical power due to discarding many study samples.
These framework is is for postprediction missing data imputation


```
source venv/bin/activate
pip install -r requirement.txt
deactivate
```

The method used to semi-supervised learning for the imputation is 

KNN,Missforest,XGBoost,NN,RidgeRegression,Kernel Approximation,Mean and median