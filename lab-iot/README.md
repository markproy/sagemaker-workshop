# Internet of Things (IoT) lab

* This lab will take advantage of an existing AWS IoT workshop that can be found [here](http://iot.awsworkshops.com/).
* Navigate to the Getting Started / EC2 page.
* Open the link in a new browser tab to create the CloudFormation stack.
    * Use your username (e.g. `user15`) as your prefix for the Stack name and the Workshop Name.
* Follow the lab instructions to create a new thing and publish and subscribe to IoT messages. At a high level, this will involve the following steps:
    * Create a new IoT thing
    * Update the IAM policy
    * Download the certificates zip file and unzip the file
    * Rename the public and private certificates to match the filenames expected
    * Upload `certificate.pem` and `privateKey.pem` to the lab1 folder in the Cloud9 environment
    * Edit the Python script in Cloud9. Change your client name to be the same as your AWS user name (e.g. `user15`). Change text message being published to include your username.
    * Run the Python publishing script to repeatedly publish messages
    * On the IoT console, subscribe to `#` (wildcard), you should see the messages arriving
* When you have completed the lab, remember to return to the CloudFormation console and delete the stack to clean up your resources.
