# Use SageMaker inference pipelines

In this lab, you will take advantage of Amazon SageMaker inference pipelines.

This makes it easy to build and deploy feature preprocessing pipelines with a suite of feature transformers available in the new SparkML and scikit-learn containers in Amazon SageMaker. You can write your data processing code once and reuse it for training and inference which provides consistency in your machine learning workflows and easier management of your models. You can deploy upto five steps in your inference pipeline and they all execute on the same instance so there is minimal latency impact. The same inference pipeline can be used for real-time and batch inferences.

The following SageMaker examples demonstrate the use of inference pipelines:

1. **Inference Pipeline with Scikit-learn and Linear Learner**, in the **SageMaker Python SDK** folder. This example shows how you can build your ML Pipeline leveraging the Sagemaker Scikit-learn container and SageMaker Linear Learner algorithm. After the model is trained, you deploy the Pipeline (Data preprocessing and Linear Learner) as an Inference Pipeline behind a single Endpoint for real time inference and for batch inferences using Amazon SageMaker Batch Transform.

2. **inference_pipeline_sparkml_xgboost_abalone**, in the **Advanced Functionality** folder. This example shows how to deploy an Inference Pipeline with SparkML for data pre-processing and XGBoost for training on the Abalone dataset. The pre-processing code is written once and used between training and inference.

3. **inference_pipeline_sparkml_blazingtext_dbpedia**, in the **Advanced Functionality** folder. The example shows how to deploy an Inference Pipeline with SparkML for data pre-processing and BlazingText for training on the DBPedia dataset. The pre-processing code is written once and used between training and inference

4. **inference_pipeline_sparkml_xgboost_car_evaluation**, in the **Advanced Functionality** folder. This example shows how to deploy an Inference Pipeline with SparkML for data pre-processing and XGBoost for training on the Car Evaluation  dataset. The pre-processing code is written once and used between training and inference

To use one of these examples, inside of your notebook instance, navigate to the **SageMaker Examples** tab. Expand the specified folder. Find the named example and click **Use**.  When prompted, click **Create Copy**. SageMaker then opens a copy of the sample notebook. Follow the detailed instructions in the notebook.

called **Introduction to Amazon Algorithms**.
