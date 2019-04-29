import boto3
#import time
import pprint
pp = pprint.PrettyPrinter(indent=3)

iamClient = boto3.client('iam')
iamResource = boto3.resource('iam')


# Define the common parameters for the users we want to create
groupName = "LM-SageMaker-Users"
group = iamResource.Group(groupName)
userString = "user"
userNumberMin = 1
userNumberMax = 26


def ListUsers() :
   print("Listing Users")
   paginator = iamClient.get_paginator('list_users')

   # For each user
   for page in paginator.paginate():

      # Print the user
      for user in page['Users']:
         #pp.pprint(user)
         print("User: {0}\nUserID: {1}\nARN: {2}\nCreatedOn: {3}\n".format(
         user['UserName'],
         user['UserId'],
         user['Arn'],
         user['CreateDate']
      )
   )


def CreateUsers() :
   print("Creating Users")

   # For each user
   for userNumber in range (userNumberMin, userNumberMax):
      userName= userString + "{0:02d}".format(userNumber)
      print("Creating: " + userName)
      iamClient.create_user(UserName=userName, Tags=[
            {'Key' : 'userNumber', 'Value' : str(userNumber) },
            {'Key' : 'workshop', 'Value' : "Liberty Mutual Sagemaker FNOL" },
            {'Key' : 'AWSContact', 'Value' : "chowdry@amazon.com" }
         ]
      )
      iamClient.create_login_profile(UserName=userName, Password=userName, PasswordResetRequired=True)
      group.add_user(UserName=userName)

def DeleteUsers() :
   print("Deleting Users")
   for userNumber in range (userNumberMin, userNumberMax):
      userName= userString + "{0:02d}".format(userNumber)
      group.remove_user(UserName=userName)
      iamClient.delete_login_profile(UserName=userName)
      iamClient.delete_user(UserName=userName)

ListUsers()

CreateUsers()

ListUsers()

#DeleteUsers()
