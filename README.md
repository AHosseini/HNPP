# Hierarchical Nonparametric Point Process (HNP3)
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

HNP3 is a python library for modeling content diffusion over social media. This library aims to infer the interests of users by jointly modeling the time and content of messages of users over social media. In order to do so, this library use a nonparametric topic model and a nonparametric point process to jointly model the time and content of messages over social media and uses an efficient online inference algorithm based on sequential monte carlo to infer the latent variables.

## Prerequisites

- Python version 3.x

## Features

-  A coherent generative model for content over social media

- The model manages its complexity by adapting the size of the latent space and the number of classifiers over time.

- Handling concept drift by adapting data-concept association without unnecessary i.i.d. assumption among data of a batch

- An online algorithm for inference on the non-conjugate non-parametric time-dependent models.

## Data

The EventRegistry dataset which is used in the HNP3 paper is in the data folder as a sample.
The dataset contians the following files:

events.csv: The events sorted in an increasing order of time in the following format:

```
<time> <doc_id> <user_id> <is_dup> <dup_doc_id> <words_len> [word_id:word_count] <ne_len> [ne_id:ne_count]
```

wordmap.pickle: The mapping between the words and their corresponding index. words indices are start from 0.

wordcount.pickle: Number of times each word have occurred. The word are represented by their corresponding index.

If the dataset contains the social relations among the users, then the dataset folder should
contain a file which contains the adjacency list among the users. 
The name of this file should be "adjacency_matrix.pickle" which is a numpy matrix that 
contains the adjacency matrix. 

## Running The Code

- Run the install script using python by ```python setup.py build_ext --inplace```

- Set the Dataset in the "Main" Script

- Run the Main script by ```python main.py MethodName DatasetName```

 where MethodName can be "HNP3" or "DirichletHawkes" or "Hawkes" and DatasetName can be "EventRegistry" or any other 
 dataset which is located in "data" folder. The results will be saved under the results folder

## Citation 

In case of using the code, please cite the following paper:

Hosseini, S.A., Khodadadi, A., Arabzadeh, A. and Rabiee H.R., 2016, December. HNP3: A Hierarchical Nonparametric Point Process for Modeling Content Diffusion over Social Media. In 2016 IEEE International Conference on Data Mining (ICDM). IEEE.
