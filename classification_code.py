#!/usr/bin/env python
# coding: utf-8

# <a href="http://cocl.us/pytorch_link_top?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork20647850-2022-01-01">
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/Pytochtop.png" width="750" alt="IBM Product " />
# </a> 
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/cc-logo-square.png" width="200" alt="cognitiveclass.ai logo" />
# 

# <h1><h1>Pre-trained-Models with PyTorch </h1>
# 

# In this lab, you will use pre-trained models to classify between the negative and positive samples; you will be provided with the dataset object. The particular pre-trained model will be resnet18; you will have three questions:
# 
# <ul>
# <li>change the output layer</li>
# <li> train the model</li> 
# <li>  identify  several  misclassified samples</li> 
#  </ul>
# You will take several screenshots of your work and share your notebook. 
# 

# <h2>Table of Contents</h2>
# 

# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <ul>
#     <li><a href="https://#download_data"> Download Data</a></li>
#     <li><a href="https://#auxiliary"> Imports and Auxiliary Functions </a></li>
#     <li><a href="https://#data_class"> Dataset Class</a></li>
#     <li><a href="https://#Question_1">Question 1</a></li>
#     <li><a href="https://#Question_2">Question 2</a></li>
#     <li><a href="https://#Question_3">Question 3</a></li>
# </ul>
# <p>Estimated Time Needed: <strong>120 min</strong></p>
#  </div>
# <hr>
# 

# <h2 id="download_data">Download Data</h2>
# 

# Download the dataset and unzip the files in your data directory, unlike the other labs, all the data will be deleted after you close  the lab, this may take some time:
# 

# In[ ]:


get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip')


# In[ ]:


get_ipython().system('unzip -q Positive_tensors.zip')


# In[ ]:


import wget
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip'
filename = wget.download(url)
print(filename)


# In[ ]:


pwd


# In[ ]:


from zipfile import ZipFile

with ZipFile('resources/Positive_tensors.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()


# In[ ]:


import wget
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip'
filename = wget.download(url)
print(filename)


# In[ ]:


from zipfile import ZipFile

with ZipFile('resources/Negative_tensors.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()


# ! wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip
# !unzip -q Negative_tensors.zip

# We will install torchvision:
# 

# In[ ]:


get_ipython().system('pip install torchvision')


# <h2 id="auxiliary">Imports and Auxiliary Functions</h2>
# 

# The following are the libraries we are going to use for this lab. The <code>torch.manual_seed()</code> is for forcing the random function to give the same number every time we try to recompile it.
# 

# In[1]:


# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)


# In[2]:


import keras


# In[3]:


from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# <!--Empty Space for separating topics-->
# 

# <h2 id="data_class">Dataset Class</h2>
# 

# This dataset class is essentially the same dataset you build in the previous section, but to speed things up, we are going to use tensors instead of jpeg images. Therefor for each iteration, you will skip the reshape step, conversion step to tensors and normalization step.
# 

# In[6]:


# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory = r"D:\Image_processing\AI_VGG_Crack\Project\resources" 
        positive = "Positive_tensors"
        negative = "Negative_tensors"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)     
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
               
        image=torch.load(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
print("done")


# In[7]:


train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
print("done")


# In[8]:


len(train_dataset)


# <h2 id="Question_1">Question 1</h2>
# 

# <b>Prepare a pre-trained resnet18 model :</b>
# 

# <b>Step 1</b>: Load the pre-trained model <code>resnet18</code> Set the parameter <code>pretrained</code> to true:
# 

# In[9]:


# Step 1: Load the pre-trained model resnet18

# Type your code here

import torchvision.models as model_res


model= model_res.resnet18(pretrained=True)


# In[ ]:





# <b>Step 2</b>: Set the attribute <code>requires_grad</code> to <code>False</code>. As a result, the parameters will not be affected by training.
# 

# In[10]:


# Step 2: Set the parameter cannot be trained for the pre-trained model


# Type your code here

for param in model.parameters():
    param.requires_grad = False


# <code>resnet18</code> is used to classify 1000 different objects; as a result, the last layer has 1000 outputs.  The 512 inputs come from the fact that the previously hidden layer has 512 outputs.
# 

# <b>Step 3</b>: Replace the output layer <code>model.fc</code> of the neural network with a <code>nn.Linear</code> object, to classify 2 different classes. For the parameters <code>in_features </code> remember the last hidden layer has 512 neurons.
# 

# In[11]:


hidden = 512
out = 2
model.fc = nn.Linear(hidden, out)  


# Print out the model in order to show whether you get the correct answer.<br> <b>(Your peer reviewer is going to mark based on what you print here.)</b>
# 

# In[12]:


print(model)


# <h2 id="Question_2">Question 2: Train the Model</h2>
# 

# In this question you will train your, model:
# 

# <b>Step 1</b>: Create a cross entropy criterion function
# 

# In[13]:


# Step 1: Create the loss function

# Type your code here

criterion = nn.CrossEntropyLoss()


# <b>Step 2</b>: Create a training loader and validation loader object, the batch size should have 100 samples each.
# 

# In[14]:


batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)


# <b>Step 3</b>: Use the following optimizer to minimize the loss
# 

# In[15]:


optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)


# <!--Empty Space for separating topics-->
# 

# **Complete the following code to calculate  the accuracy on the validation data for one epoch; this should take about 45 minutes. Make sure you calculate the accuracy on the validation data.**
# 

# In[16]:


n_epochs = 1
loss_list = []
accuracy_list = []
accuracy = 0
correct = 0
N_test = len(validation_dataset)
N_train = len(train_dataset)
start_time = time.time()
Loss= 0
result_txt=[]

for epoch in range(n_epochs):
    for i, (x, y) in enumerate(train_loader):  
        print("**********")
        print('Epoch#{}'.format(i+1))
        # set model to train 
        model.train() 
        
        # clear gradient 
        optimizer.zero_grad()
     
        # make a prediction 
        yhat = model(x)
   
        # calculate loss 
        loss = criterion(yhat, y) 
        # loss.requires_grad = True
    
        # calculate gradients of parameters 
        loss.backward()
        
        # update parameters 
        optimizer.step()
        
        loss_list.append(loss.item())
        #print(type(loss_list))
        print("Finished in {} (s)".format(time.time()- start_time))
        print("**********")
    # end for
        
    correct=0
    for i, (x_test, y_test) in enumerate(validation_loader):
        i_start_time = time.time()
        
        
        # set model to eval 
        model.eval()
       
        # make a prediction 
        yhat_val = model(x_test)
        
        # find max 
        _, yhat_val_max = torch.max(yhat_val.data, 1)
       
       
        #Calculate misclassified  samples in mini-batch 
        #hint +=(yhat==y_test).sum().item()
        correct += (yhat_val_max==y_test).sum().item()  
        
        print("Finished in {} (s)".format(time.time()-start_time))
        


# end for


# In[ ]:





# In[17]:


# end for

accuracy=correct/N_test
print("Epoch %d - accuracy: %.3f" % (epoch+1, accuracy))


# In[18]:


pwd


# In[ ]:


loss_list


# In[19]:


# open file in write mode
with open(r'resources/losses.txt', 'w') as fp:
    for loss in loss_list:
        # write each item on a new line
        fp.write("%s\n" % loss)
    print('Done')


# In[20]:


type(loss_list)


# <b>Print out the Accuracy and plot the loss stored in the list <code>loss_list</code> for every iteration and take a screen shot.</b>
# 

# In[21]:


accuracy


# In[1]:


import matplotlib.pyplot as plt
with open('resources/losses.txt', 'r') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]



# In[2]:


x[0:5]


# In[3]:


from matplotlib.pyplot import figure

figure(figsize=(12, 5), dpi=80)

plt.title("Loss")
plt.plot(x)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()


# <h2 id="Question_3">Question 3:Find the misclassified samples</h2> 
# 

# <b>Identify the first four misclassified samples using the validation data:</b>
# 

# In[38]:


count = 0
max_num_of_items = 4  # first four mis-classified samples
validation_loader_misclassified = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1)

for i, (x_test, y_test) in enumerate(validation_loader_misclassified):
    # set model to eval
    model.eval()
    
    # make a prediction
    yhat = model(x_test)
    
    # find max
    _, yhat_max = torch.max(yhat.data, 1)
    
    # print mis-classified samples
    if yhat_max != y_test:
        print("Sample: {}; Predicted value: {}; Actual value: {}".format(str(i), str(yhat_max), str(y_test)))
        count += 1
        if count >= max_num_of_items:
            break
    # end if
# end for   


# In[36]:


pwd


# In[ ]:





# <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/share-notebooks.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork20647850-2022-01-01"> CLICK HERE </a> Click here to see how to share your notebook.
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork20647850-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
# | ----------------- | ------- | ---------- | ----------------------------------------------------------- |
# | 2020-09-21        | 2.0     | Shubham    | Migrated Lab to Markdown and added to course repo in GitLab |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 

# Copyright © 2018 <a href="https://cognitiveclass.ai/?utm_medium=dswb&utm_source=bducopyrightlink&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork20647850-2022-01-01&utm_campaign=bdu">cognitiveclass.ai</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork20647850-2022-01-01">MIT License</a>.
# 
