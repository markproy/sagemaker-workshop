# Hello world, Amazon SageMaker Feature Store
This notebook provides a demo of how easy it is to use SageMaker Feature Store. It does this by leveraging a simple set of utility functions that wrap the feature store API to keep it simple for a data scientist using Python.

You'll find a notebook that shows:

- Creating a new feature group based on column definitions in a Pandas dataframe
- Deleting existing feature groups, including offline store data
- Ingesting data directly from a Pandas dataframe
- Extracting the latest values from the online store into a Python friendly dictionary
- Extracting training datasets without knowing anything about Athena or Glue
- Building an ML model directly from a dataset extracted from the feature store
- Using time travel to extract point-in-time correct data based on a specific timestamp
- Deleting records and showing the impact of that deletion
- Getting a count of offline store records for a feature group
- Saving and retrieving tags for feature groups, including handling of documentation URL's
- Download and view a sample parquet file from an offline store
