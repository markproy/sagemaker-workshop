# Preparing accounts for the workshop

Before hosting this workshop for a customer, an account needs to be prepared. If the customer is new to AWS, or is not already using Amazon SageMaker, we often create a new AWS account on their behalf. When attendees already have accounts with access to SageMaker, we can instead ensure the proper account limits are in place. When in doubt, we provide the account.

Once the account is identified or created, account limits need to be confirmed or raised to support the workshop. By default, we use `us-east-1` as the region.

## Account limits

The default account limits related to SageMaker are typically set to support minimal usage by a single user. When supporting a multi-user workshop from a single account, it is important to confirm or raise the limits to at least those numbers described here. Wherever a *number of users* is referenced, insert a number with an additional buffer to account for  last minute attendees. For a group of 15 planned users for example, use 20 as the number of users when setting these limits.

1. **Notebooks**. Set `ml.c5.xlarge` to number of users. Set `ml.t3.medium` to number of users. Set `total_count` to number of users times 2. Set `total_volume_size_in_gb` to 100,000.

2. **Training**. Set `instance_per_job_max` to 20. Set `ml.c5.xlarge` to number of users times 3. Set `ml.m4.xlarge` to number of users times 3. Set `ml.p2.xlarge` and `ml.p3.xlarge` to number of users (required if you would like to do any computer vision labs with SageMaker built-in algorithms). Set `total_instance_count` to number of users times 3.

3. **Hosting**. Set `instance_per_endpoint_max` to 4. Set `ml.c5.xlarge` to number of users. Set `ml.eia1.medium` to number of users. Set `ml.m4.xlarge` to number of users. Set `total_instance_count` to number of users times 3.

4. **Batch transform**. Set `ml.c5.xlarge` and `ml.m4.xlarge` to number of users. Set `total_instance_count` to number of users times 2.

5. **Automatic Model Tuning (HPO)**. Set `max_parallel_training_jobs` to 10. Set `max_training_jobs` to number of users times 20. Set `total_tuning_jobs` to number of users times 30.

6. **Ground Truth**. Set `max_dataset_objects_per_labeling_job` to 10,000. Set `total_labeling_jobs` to number of users times 2.

7. **Compilation jobs**. Set `max_parallel_compilation_jobs` to the number of users.


Customers can request account limit increases via their support console. Internally, we can use `https://sagemaker-tools.corp.amazon.com/limits`.

## SageMaker API throttles

When supporting a SageMaker workshop for many users under a single account, there will be times when many users are all doing the same thing in the SageMaker console at the same exact time. There are API throttles in place to avoid over-taxing the service for cases of mistaken or malicious behavior. To avoid disrupting the attendee experience with errors in the console, it is important to ensure throttle limits are raised accordingly.

- **Operations** = ListNotebookInstances, ListTrainingJobs, ListLabelingJobs, ListHyperParameterTuningJobs, ListModels, ListEndpoints, ListTransformJobs,
CreateNotebookInstance, StartNotebookInstance, CreateTrainingJob,
CreateEndpoint, DeleteEndpoint, ListEndpointConfigurations, ListEndpoints

- **Rate** = 5

- **Burst** = number of users

An internal trouble ticket should be opened to AWS /  SageMaker / Service Limit increase. It should specify the account, the region, and the above burst and rate limits. Specify a justification such as `We are hosting a SageMaker workshop for up to NN customers from DATE1 to DATE2. All users will be working from the same account.`

## SageMaker initial execution role

To avoid confusion and bumping into other permission challenges, it can be helpful to establish the first SageMaker IAM execution role for the account before the workshop. To do so:

- Sign in to AWS in the account as an admin user.
- Begin to create the first notebook instance in the account.
- When prompted for an execution role, choose to create a brand new role.
- Choose to allow the notebook to have access to `Any S3 bucket`.
- This avoids having to use `sagemaker` in the bucket name. This also avoids the users running into an `access denied` error when uploading data.

## Account creation

The account should be created with a group (e.g., "attendees")  containing multiple users (e.g., "user01" through "user25" for a projected 20-person workshop). Users will need both console and programmatic access. They will need SageMakerFullAccess and S3FullAccess permissions, and an admin user should be provided for the workshop leader.
