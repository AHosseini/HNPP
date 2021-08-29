# Hierarchical Nonparametric Point Process (HNPP)
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

Hierarchical NonParmetric Point Processes (HNPP) is unified statistical framework for modeling temporal marked events and predicting future events using the history of events. HNPP is able to model and infer the growing number of patterns underly the events by sharing the patterns among users using a hierarchical structure. In this library we implemented two variants of HNPP, i.e. Hierarchical Dirichlet Point Process (HDPP) and Factorized Gamma Point Processes (FGPP). HDPP models the time of events using a multi-dimensional point process and mark of events by a Dirichlet Process and shares the patterns among users by a common base measure. FGPP  is another variant of HNP3 which infers the latent patterns underly events and shares such patterns among users using a size-biased ordering. In this library, we implemented an online inference algorithm based on sequential Monte Carlo for HDPP which is implemented in Python and a scalable variational inference algorithm for FGPP which is implemented in Matlab.

## Datasets
In order to evaluate the HDPP method in the context of content diffusion over social networks, we gathered two datasets from [EventRegistry](http://eventregistry.org/) and Twitter. These datasets can be downloaded from [here](https://drive.google.com/file/d/1VeeFNc1bDY2D_7TrnkGecZ9aD8MLbKdi/view?usp=sharing) and [here](https://drive.google.com/file/d/12j5hBSm3PuFgvGs7FYQ9cB5dO2H1eIVM/view?usp=sharing) respectively. In order to run the code on these datasets, copy these files in data folder of HDPP folder.

To evaluate the FGPP, we used the [Tianchi](https://drive.google.com/file/d/177UrNudhNI97CS8Y-a3UEzeMFqFQm1gn/view?usp=sharing) and [last.fm](https://drive.google.com/file/d/1fPMoO-HFXSqOses1ClSWJMbjbQA5boZK/view?usp=sharing) datasets. In order to run the code on these datasets, download these datasets and copy these files in Datasets folder of FGPP folder.

## Prerequisites

- Python version 3.x

- Matlab version R2014 or later

## Features

-  A coherent generative model for content over social media

-  A novel generative model for learning user preferences and recommendation

- The models manage their complexity by adapting the size of the latent space over time.

- An online algorithm for inference on HDPP and a scalable variational inference algorithm for FGPP.

## Data

The EventRegistry and Twitter dataset which is used for evaluating the HDPP model is in the data folder as a sample.
For HDPP model, the dataset should contain the following files:

events.csv: The events sorted in an increasing order of time in the following format:

```
<time> <doc_id> <user_id> <is_dup> <dup_doc_id> <words_len> [word_id:word_count] <ne_len> [ne_id:ne_count]
```

wordmap.pickle: The mapping between the words and their corresponding index. Indices start from 0.

wordcount.pickle: Number of times each word have occurred. The word are represented by their corresponding index.

nemap.pickle: The mapping between the named entities and their corresponding index. Indices start from 0.

necount.pickle: Number of times each named entity have occurred. The word are represented by their corresponding index.

If the dataset contains the social relations among the users, then the dataset folder should
contain a file which contains the adjacency list among the users. 
The name of this file should be "adjacency_matrix.pickle" which is a numpy matrix that 
contains the adjacency matrix. 

## Running The Code

In order to run HDPP model:

- Run the install script using python by ```python setup.py build_ext --inplace```

- Set the Dataset in the "Main" Script

- Run the Main script by ```python main.py MethodName DatasetName```

 where MethodName can be "HDPP" or "DirichletHawkes" or "Hawkes" and DatasetName can be "EventRegistry" or any other 
 dataset which is located in "data" folder. The results will be saved under the results folder.
 
 In order to run FGPP method:

Go to the methods folder

Set the Dataset in the "RunFGPP" Script

Run the run script

The results will be saved under the "Results" folder.
 

## Citation 

In case of using the code, please cite the following paper:

Hosseini, S.A., Khodadadi, A., Arabzadeh, A. and Rabiee H.R., 2016, December. HNP3: A Hierarchical Nonparametric Point Process for Modeling Content Diffusion over Social Media. In 2016 IEEE International Conference on Data Mining (ICDM). IEEE.
