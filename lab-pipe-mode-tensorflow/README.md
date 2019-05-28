This sample notebook demonstrates the following combination:

1.	**Pipe mode** – for scaling the amount of training data infinitely, and not having to wait for a full data download to begin training
2.	**TensorFlow script mode** – for easy use of the SageMaker training service with your own TensorFlow code (no custom Docker containers needed)
3.	**Multiple training files** – given your scale, you will want to point to a channel with a set of files, each containing sets of samples
4.	**TFRecords** – native format for working with TensorFlow efficiently
5.	**Multiple channels** – training, testing, and validation

The notebook lets you configure the number of features, number of samples, and the number of files per channel. It also lets you configure the batch size and number of epochs. Lastly, it can run in either File mode or Pipe mode so that you can experiment with either.
