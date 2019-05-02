# Use the built-in BlazingText algorithm to perform text classification

In this lab, you will train and use a model to perform text classification,
taking advantage of the built-in BlazingText algorithm in SageMaker. The use case is for predicting one of 14 non-overlapping classes of articles from Wikipedia. The model will be built using 560,000 training samples.

Here are some notes for completing the lab:

1. Inside of your notebook instance, navigate to the **SageMaker Examples** tab. Expand the folder
called **Introduction to Amazon Algorithms**. Find the line called **blazingtext_text_classification_dbpedia**, and
click **Use** and **Create copy**.

2. Click on the **Files** tab and open the new folder created. Click on the Jupyter notebook to start the lab.

3. Navigate to the notebook and follow the documentation provided in the sample notebook.

4. You should add your initials to the `prefix` in the first cell (e.g., `mpr_blazingtext/supervised` instead of just `blazingtext/supervised`). This will make it easier to find your files, if you will be sharing the same S3 bucket with other workshop users.

5. Since we have multiple users in the same account, we want to make it easier to find your training jobs and endpoints as well. To faciliate this, locate the cell that creates the `Estimator`, and add a `base_job_name` parameter to the call. It should look like the following, except you would replace `mpr` with **your** initials or user name. Note that you will also need to specify `ml.c5.xlarge` as the training instance type to align with the account limits that were preset for the workshop.

``
bt_model = sagemaker.estimator.Estimator(container,
  role, train_instance_count=1,
   train_instance_type='ml.c5.xlarge',
   train_volume_size = 30,
   train_max_run = 360000,
   input_mode= 'File',
   output_path=s3_output_location,
   base_job_name='mpr-bt',
   sagemaker_session=sess)
``
