# Use SageMaker inference pipelines

In this lab, you will take advantage of Amazon SageMaker inference pipelines. Many real world use cases require a more complicated set of steps including pre-processing and post-processing that should ideally be shared across training and prediction. SageMaker inference pipelines provides a mechanism for exactly this purpose.

The following SageMaker examples demonstrate the use of inference pipelines:

1. **Inference Pipeline with Scikit-learn and Linear Learner**, in the **SageMaker Python SDK** folder. This example shows how you can build your ML Pipeline leveraging the Sagemaker Scikit-learn container and SageMaker Linear Learner algorithm. After the model is trained, you deploy the Pipeline (Data preprocessing and Linear Learner) as an Inference Pipeline behind a single Endpoint for real time inference and for batch inferences using Amazon SageMaker Batch Transform.

2. **inference_pipeline_sparkml_xgboost_abalone**, in the **Advanced Functionality** folder. This example shows how to deploy an Inference Pipeline with SparkML for data pre-processing and XGBoost for training on the Abalone dataset. The pre-processing code is written once and used between training and inference.

3. **inference_pipeline_sparkml_blazingtext_dbpedia**, in the **Advanced Functionality** folder. The example shows how to deploy an Inference Pipeline with SparkML for data pre-processing and BlazingText for training on the DBPedia dataset. The pre-processing code is written once and used between training and inference

4. **inference_pipeline_sparkml_xgboost_car_evaluation**, in the **Advanced Functionality** folder. This example shows how to deploy an Inference Pipeline with SparkML for data pre-processing and XGBoost for training on the Car Evaluation  dataset. The pre-processing code is written once and used between training and inference

To use one of these examples, inside of your notebook instance, navigate to the **SageMaker Examples** tab. Expand the specified folder. Find the named example and click **Use**.  When prompted, click **Create Copy**. SageMaker then opens a copy of the sample notebook. Follow the detailed instructions in the notebook.

called **Introduction to Amazon Algorithms**.
