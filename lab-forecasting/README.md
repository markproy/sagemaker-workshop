# Forecasting lab

Many customers have turned to machine learning and deep learning to get more accurate time series forecasts across a wide range of use cases (product demand, store inventory, temperatures, ...). AWS provides solutions or building blocks at multiple levels of its stack.

At the top level, Amazon Forecast is a new services that is in preview at the time of this writing. Users of this service do not need any ML or deep learning expertise.

At the next level down in the stack, Amazon SageMaker provides a built-in algorithm called [DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html). Here are some additional resources to help you get started:

- See this [blog post](https://aws.amazon.com/blogs/machine-learning/now-available-in-amazon-sagemaker-deepar-algorithm-for-more-accurate-time-series-forecasting/) for a walkthrough.

- Here is [another blog post](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-deepar-now-supports-missing-values-categorical-and-time-series-features-and-generalized-frequencies/) covering some new features such as support for missing values, categorical time series, and generalized frequencies.

- Lastly, here is a [blog post](https://medium.com/@julsimon/predicting-world-temperature-with-time-series-and-deepar-on-amazon-sagemaker-e371cf94ddb5) showing the use of DeepAR to predict temperatures.

In this lab, you will exercise the DeepAR built-in algorithm on time series electricity data. We will use a SageMaker-provided sample notebook called `deepar_synthetic`, which you can find in the Introduction to Amazon Algorithms section. Use a copy of that sample, and follow its detailed instructions.

Be sure to try out the interactive display of the forecasts predicted by DeepAR for a given customer on a given forecast day. The notebook first demonstrates base forecasting capability, and then moves on to show how it can handle missing data and special days (e.g., promotion days in a retail setting).

NOTE: the notebook defaults to using `ml.c4.2xlarge`. If you hit an account limit when training, you should switch to `ml.c5.xlarge` instead.
