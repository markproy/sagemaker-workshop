# Perform bulk inference in batch

There are many use cases that do not require a real-time prediction endpoint, but are more effectively performed in a batch against a large set of observations. In this lab, you will use a model in batch. For more details, see [this](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html) documentation.

We will use a SageMaker-provided sample notebook called `batch_transform_pca_dbscan_movie_clusters.ipynb`. You can find it in the `Advanced functionality` section of the examples tab in your notebook. Use a copy of that sample, and follow its detailed instructions.

**NOTE:** Skip the `Clustering (optional)` section at the bottom of the notebook.
