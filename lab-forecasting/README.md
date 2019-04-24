# Forecasting lab

Many customers have turned to machine learning and deep learning to get more accurate time series forecasts across a wide range of use cases (product demand, store inventory, temperatures, ...). AWS provides solutions or building blocks at multiple levels of its stack.

At the top level, Amazon Forecast is a new services that is in preview at the time of this writing. Users of this service do not need any ML or deep learning expertise.

At the next level down in the stack, Amazon SageMaker provides a built-in algorithm called [DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html). See this [blog post](https://aws.amazon.com/blogs/machine-learning/now-available-in-amazon-sagemaker-deepar-algorithm-for-more-accurate-time-series-forecasting/) for a walkthrough.

In this lab, you will exercise the DeepAR built-in algorithm on time series electricity data. We will use a SageMaker-provided sample notebook called `deepar_synthetic`, which you can find in the Introduction to Amazon Algorithms section. Use a copy of that sample, and follow its detailed instructions.
