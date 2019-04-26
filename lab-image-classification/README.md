# Use built-in Image Classification algorithm

In this lab, you will train and use a model to do image classification,
taking advantage of the built-in Image Classification algorithm in Amazon SageMaker. The use case is for detecting
objects from a set of 256 types.

This notebook requires your account to have access to a GPU instance type such
as `ml.p3.2xlarge`. Confirm with your workshop host that your account limits have been adjusted accordingly.

1. Inside of your notebook instance, navigate to the **SageMaker Examples** tab. Expand the folder
called **Introduction to Amazon Algorithms**. Find the line called **Image-classification-fulltraining-highlevel** and
click **Use**.

2. When prompted, click **Create Copy**. SageMaker then opens a copy of the sample notebook.

3. Follow the detailed instructions provided in the sample notebook. For best results,  
use transfer learning by changing the following hyperparameter: `use_pretrained_model=1`. Also, to reduce training time, only save checkpoints at the end of the training job: `checkpoint_frequency=5`.
