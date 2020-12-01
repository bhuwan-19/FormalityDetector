## TextFormality

## Overview

This project is to estimate the formality of the text and classify the result into 5 categories, more exactly, 
give the category marks between 1 and 5. In this project, the Gensim, NLTK, Spacy frameworks are used to estimate the 
formality.
Also, a pre-trained model and word2vec pre-trained vectors are used for Natural Language Processing.

## Structure

- src

    The main source code for pre processing the text, extraction of their features, and calculation of its formality.
    
- utils

    * The pre-trained models for NLP
    * The source code for management of the folders and files in this project
    
- app

    The main execution file

- requirements

    All the dependencies for this project
    
- settings

    Several settings including the model path and some correlation coefficients

## Installation

- Environment

    Ubuntu 18.04, Windows 10, Python 3.6

- Dependency Installation

    Please go ahead to this project directory and run the following commands in the terminal
    ```
        pip3 install -r requirements.txt
        python3 -m nltk.downloader wordnet
    ```
 
## Execution

- Please run the following command in the terminal

    ```
        python3 app.py
    ```
