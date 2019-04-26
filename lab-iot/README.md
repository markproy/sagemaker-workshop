# Internet of Things (IoT) lab

* This lab will take advantage of an existing AWS IoT workshop that can be found [here](http://iot18.awsworkshops.com/basics/).
* Navigate to the Getting Started / EC2 page.
* Open the link in a new browser tab to create the CloudFormation stack.
    * Use your username as your prefix.
* You may want to disconnect from your VPN to avoid getting blocked.
* Follow the instructions to create a new thing and publish and subscribe to IoT messages. At a high level, this will involve the following steps:
    * Create a new IoT thing
    * Update the IAM policy
    * Download the certificates zip file and unzip the file
    * Rename the certificates to match the filenames expected
    * Upload `certificate.pem` and `privateKey.pem` to the Cloud9 environment
    * Edit the Python script in Cloud9. Change your topic name to be the same as your AWS user name. Change one of the values in your message to be your username.
    * Run the Python publishing script to repeatedly publish some simple messages.
    * On the IoT console, subscribe to your topic and you should see the messages arriving.
* When you have completed the lab, remember to return to the CloudFormation console and delete the stack to clean up your resources.
