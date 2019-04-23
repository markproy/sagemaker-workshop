# Amazon SageMaker Workshop
This workshop provides a set of labs designed to give you hands-on experience building, training, and deploying machine
learning models using Amazon SageMaker. The target audience includes data scientists, machine learning engineers, solutions architects, and software developers.

The workshop can be used in self-paced fashion, or delivered by a solutions architect in a 1-day or 2-day format.

## Lab content

The "setup lab" is provided for those that have never created their own SageMaker notebook instance. Once that is complete, the following labs are available:

- creating your [**first notebook instance**](lab-0-setup/README.md) ("setup" lab)
- [**logistic regression**](lab-xgboost/README.md) using the built-in XGBoost algorithm
- [**image classification**](lab-image-classification) using the built-in Image Classification algorithm
- [**bring your own neural network script**](lab-bring-your-own-tensorflow/README.md) to a container provided by Amazon SageMaker
- perform [**hyperparameter optimization**](lab-hpo/README.md)
- perform [**batch inference**](lab-batch-inference/README.md)
- perform [**A/B testing**](lab-ab-testing/README.md) when deploying a new version of an existing hosted model
- use [**auto-scaling**](lab-auto-scale/README.md) to improve scalability of an endpoint hosted by Amazon SageMaker
- [**bring your own Docker container**](lab-bring-your-own-container/README.md) to Amazon SageMaker
- [**use inference pipelines**](lab-inference-pipelines/README.md) to build and deploy feature preprocessing pipelines and reuse them for training and inference
