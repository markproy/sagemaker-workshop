# Amazon SageMaker Workshop
This workshop provides a set of labs designed to give you hands-on experience building, training, and deploying machine
learning models using Amazon SageMaker. The target audience includes data scientists, machine learning engineers, solutions architects, and software developers.

The workshop can be used in self-paced fashion, or delivered by a solutions architect in a 1-day or 2-day format.

## Lab content

The following labs are provided:

- create your [**first notebook instance**](lab-0-setup/README.md) ("setup" lab), which is a prerequisite for the other labs
- explore [**logistic regression**](lab-xgboost/README.md) using SageMaker's built-in XGBoost algorithm
- [**classify images**](lab-image-classification) using SageMaker's built-in Image Classification algorithm with a domain of 256 classes of objects (horse, kayak, teapot, ...)
- use DeepAR, one of SageMaker's built-in algorithms, to [**perform forecasting**](lab-forecasting) of electricity demand
- try out [**text classification**](lab-text-classification) using SageMaker's built-in BlazingText algorithm
- [**bring your own neural network script**](lab-bring-your-own-tensorflow/README.md) to a container provided by Amazon SageMaker
- perform [**hyperparameter optimization**](lab-hpo/README.md)
- perform [**batch inference**](lab-batch-inference/README.md) to get predictions on a large number of observations in bulk
- perform [**A/B testing**](lab-ab-testing/README.md) when deploying a new version of an existing model hosted by Amazon SageMaker
- use [**auto-scaling**](lab-auto-scale/README.md) to improve scalability of an endpoint hosted by Amazon SageMaker
- [**use inference pipelines**](lab-inference-pipelines/README.md) to build and deploy feature preprocessing pipelines and reuse them for training and inference
- [**bring your own Docker container**](lab-bring-your-own-container/README.md) to Amazon SageMaker
- try the [**Amazon Textract service**](lab-textract), demonstrating how it can be used to identify headers and footers
