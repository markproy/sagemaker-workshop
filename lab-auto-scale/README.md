# Use auto-scaling on a SageMaker-hosted endpoint

In this lab, you will learn how to add auto-scaling to your Amazon SageMaker hosted endpoint. This lab has a dependency on the [bring your own script](../lab-bring-your-own-tensorflow/README.md) lab. It uses the model and endpoint you produced in that lab, demonstrating how SageMaker will expand the cluster. Scaling is based on a  policy that ensures nodes in the cluster are not exceeding a threshold of invocations per minute.

In the [setup lab](../lab-0-setup/README.md), you cloned the workshop GitHub repository. Navigate to the Jupyter notebook provided for this lab in the `lab-auto-scale` folder. Follow the detailed instructions to perform the lab.

As an extra credit assignment, complete the transition to the second version of the model by eliminating the v1 variant and changing the v2 variant weight to 100%.
