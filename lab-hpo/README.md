# Perform hyperparameter optimization

In this lab, you perform

You will leverage a notebook from `SageMaker Examples`. Expand the section called `Hyperparameter Tuning` and use two notebooks:

1. `hpo_xgboost_direct_marketing_sagemaker_python_sdk.ipynb`. To reduce the time to complete this lab, please change the total number of jobs to 12 instead of 20. You will find this just before the `Launch Hyperparameter Tuning` section of the notebook. You simply update one parameter when constructing the `HyperparameterTuner` object, as follows: `max_jobs=12`
2. `HPO_Analyze_TuningJob_Results.ipynb`. After completing the first notebook, this second notebook lets you analyze the results of the HPO training jobs.

Follow the detailed instructions in the notebooks (in the order above) to complete the lab.
