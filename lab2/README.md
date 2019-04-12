# Use built-in Image Classification algorithm
In this lab, you will train and use a model to do image classification, 
taking advantage of the built-in Image Classification algorithm in SageMaker. The use case is for detecting
objects from a set of 256 types. This notebook requires your account to have access to a GPU instance type such
as ml.p3.2xlarge. Confirm with your workshop host that your account limits have been adjusted accordingly.

Inside of your notebook instance, navigate to the **SageMaker Examples** tab. Expand the folder 
called **Introduction to Amazon Algorithms**. Find the line called **Image-classification-fulltraining-highlevel** and
click **Use**.

Click on the **Files** tab and open the new folder created. Click on the Jupyter notebook called 
**Image-classification-fulltraining-highlevel.ipynb** to start the lab.

Navigate to the notebook and follow the documentation provided in the sample notebook. Be sure to increase the
number of epochs if you would like to have the test image classified properly. The notebook defaults to 5, but
25 epochs is more appropriate.
