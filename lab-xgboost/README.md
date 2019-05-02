# Use built-in XGBoost algorithm to perform logistic regression

In this lab, you will train and use a model to perform logistic regression,
taking advantage of the built-in XGBoost algorithm in SageMaker. The use case is for predicting
customer churn for a mobile phone company.

Here are some notes for completing the lab:

1. Inside of your notebook instance, navigate to the **SageMaker Examples** tab. Expand the folder
called **Introduction to Applying Machine Learning**. Find the line called **xgboost_customer_churn**,and click **Use** and **Create copy**.

2. Click on the **Files** tab and open the new folder created. Click on the Jupyter notebook to start the lab.

3. Navigate to the notebook and follow the documentation provided in the sample notebook.

4. Be sure to change the S3 bucket name in the first cell.

5. Since we have multiple users in the same account, we want to make it easier to find your training jobs and endpoints. Locate the cell that launches the training job using the `xgb.fit` method. Add a `base_job_name` parameter when creating the SageMaker Estimator object. It should look like the following, and you would replace `mpr` with your initials or user name.

``
xgb = sagemaker.estimator.Estimator(container,
    role, train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    base_job_name='mpr-xgboost',
    sagemaker_session=sess)
``

NOTE: Some browsers have had trouble displaying some parts of the notebook documentation. If the documentation on `Assigning Costs` does not display correctly, try viewing the documentation of the [notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn.ipynb )
in github.
