# Perform hyperparameter optimization

In this lab, you will work with Amazon SageMaker's [automatic model tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html) capability, also known as hyperparameter tuning or optimization. Here is a [blog post](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/) outlining how model tuning fits in the overall machine learning process, and walking you through an example.

The lab leverages a sample notebook from `SageMaker Examples`. Expand the section called `Hyperparameter Tuning` and use two notebooks:

1. `hpo_xgboost_direct_marketing_sagemaker_python_sdk.ipynb`. To reduce the time to complete this lab, please change the total number of jobs to 12 instead of 20. You will find this just before the `Launch Hyperparameter Tuning` section of the notebook. You simply update one parameter when constructing the `HyperparameterTuner` object, as follows: `max_jobs=12`. This first notebook ends up launching the tuning job, which in turn is launching a set of training jobs. The next notebook helps you analyze the results. Note the  tuning job name from this first notebook before proceeding to the second one.

2. `HPO_Analyze_TuningJob_Results.ipynb`. After completing the first notebook, this second notebook lets you analyze the results of the HPO training jobs. You can track the training jobs as they are completing, and see how it narrows in on the best hyperparameters over time. You will also assess how closely correlated each hyperparameter was to the target objective.

Follow the detailed instructions in the notebooks (in the order above) to complete the lab.
