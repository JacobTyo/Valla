# Valla

**Valla**: A standardized benchmark for authorship attribution and verification. 

Paper: [On the State of the Art in Authorship Attribution and Authorship Verification](https://arxiv.org/abs/2209.06869)

The name was chosen in memory of Lorenzo Valla, who in 1440, 
published [*De falso credita et ementita Constantini Donatione declamatio*](https://www.taylorfrancis.com/chapters/edit/10.4324/9780203816653-15/discourse-forgery-alleged-donation-constantine-lorenzo-valla-1406%E2%80%931457), 
which proved that the *Donation of Constantine* 
(where Constantine I gave the whole of the Western Roman Empire to the Roman Catholic Church) was a forgery, 
using word choice and other vernacular stylistic choices as evidence. 

## Installation

The requirements for this project were managed with conda. See the `environment.yml` file for more information.
Note that `environment_torched_adhom.yml` is an environment specifically for 
working with `authoridentification/methods/torched_adhominem.py`.

## Datasets

To use any of the datasets in this repository, first download the dataset according to the instructions 
at the top of the corresponding script. The project expects the data to be in a structure like: 
```
datasets
└───Dataset1
│   └───Raw
│   │   │   Raw Files
│   │   │   ...
│   │   
│   └───Processed
│   │   │   Processed Files will be saved here
│   │   │   ...
└───Dataset2
|   ...
```
Once the data is placed and extracted into the corrsponding Raw directory, the scripts can process them. 
The datasets currently supported are:

1. Amazon
2. Blogs
3. CCAT50
4. CMCC
5. Guardian
6. Gutenberg
7. IMDB
8. PAN20
9. PAN21
10. Reddit
11. TopicConfusion

## Methods

This repository holds implementations of several popular Authorship Attribution and Verification methodologies,
including some based on BERT, Siamese Models, Multi-Headed Language models, BiLSTMs, compression models, Ngrams, and more.
After processing a raw dataset with the corresponding script, the dataset is ready for use with any of these models. 
See each file in `valla/methods` for more information on the available methods. 

## Logging and Hyperparameter Tuning

This project uses [Weights & Biases](https://docs.wandb.ai/) both for logging and hyperparameter sweeps.

## Cite

If you use this software, place cite our paper: [On the State of the Art in Authorship Attribution and Authorship Verification](https://arxiv.org/abs/2209.06869)

## Contact

Feel free to contribute, and drop me a note at `jacob.tyo@gmail.com` if you have any questions/comments/concerns.
