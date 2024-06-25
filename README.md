# Query-Auto-Completion-


This repository contains a Python implementation of a Query Auto-Completion (QAC) model using bigram and trigram probabilities. The model is trained on a dataset consisting of various text files and predicts the next characters based on the context provided. The model can generate text by predicting the next possible characters iteratively.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Data](#data)
- [Training](#training)
- [Generating Text](#generating-text)


## Introduction

This repository implements a query auto-completion model that uses bigram and trigram probabilities to predict the next characters based on the given context. The model can generate text by iteratively predicting the next possible characters.

## Installation

To run the code, you need to have Python 3.x installed along with the following libraries:
- `requests`
- `numpy`

## Usage
The code and detailed usage instructions are provided in the [Usage.ipynb](Usage.ipynb) Jupyter Notebook.

## Model Details

The Query Auto-Completion (QAC) model implemented in this repository uses a combination of bigram and trigram probabilities to predict the next characters in a given text sequence. Here's a detailed breakdown of how the model works:

### N-Grams

- **Bigram**: A bigram is a sequence of two adjacent elements from a string of tokens. For example, in the phrase "hello world," the bigrams are "he," "el," "ll," "lo," "o " (space), " w," "wo," "or," "rl," "ld."
- **Trigram**: Similarly, a trigram is a sequence of three adjacent elements from a string of tokens. In the same phrase, the trigrams are "hel," "ell," "llo," "lo " (space), "o w," " wo," "wor," "orl," "rld."

### Data Preprocessing

1. **Text Normalization**: The input text data is converted to lowercase to ensure uniformity.
2. **Token Insertion**: Special tokens `<s>` and `</s>` are inserted at the beginning and end of the text sequences to denote the start and end of a sequence respectively.
3. **Character Set Creation**: A set of unique characters from the dataset is created, which includes all characters that appear in the text as well as the special tokens `<s>` and `</s>`.

### Probability Calculation

- **Bigram Frequency**: The model counts the occurrences of each bigram in the dataset.
- **Trigram Frequency**: Similarly, the model counts the occurrences of each trigram.
- **Probability Table**: A probability table is constructed where each entry (i, j) represents the probability of the j-th character given the i-th bigram. This is calculated using the formula:
  
  \[
  P(c_k | b_{ij}) = \frac{\text{count}(b_{ij}c_k) + 1}{\text{count}(b_{ij}) + \text{num_chars}}
  \]

  where \( b_{ij} \) is a bigram, \( c_k \) is a character, and `num_chars` is the total number of unique characters.

### Text Generation

The text generation process involves the following steps:

1. **Initialization**: The generation starts with an initial context (e.g., "he").
2. **Bigram Context**: The last two characters of the current context form the bigram used to predict the next character.
3. **Next Character Prediction**: Based on the bigram context, the model uses the probability table to predict the next character. The top-k probable next characters are considered, and one is selected based on the probabilities.
4. **Iteration**: The predicted character is appended to the context, and the process is repeated until the desired text length is reached or a terminating character (like `</s>`) is predicted.

### Advantages and Limitations

- **Advantages**:
  - Simplicity: The model is simple to understand and implement.
  - Efficiency: Using bigram and trigram probabilities allows for efficient text generation.

- **Limitations**:
  - Limited Context: The model only considers the last two characters (bigram) for predicting the next character, which might not capture long-range dependencies in the text.
  - Data Sparsity: The model's performance heavily relies on the availability of bigram and trigram counts in the training data. Rare bigrams and trigrams may lead to less accurate predictions.

Overall, this bigram and trigram-based QAC model provides a straightforward yet effective approach to text auto-completion, making it a valuable tool for applications like query suggestions and text generation.


## Data

The dataset used for training the Query Auto-Completion (QAC) model consists of Shakespearian text data. The data provides a rich source of diverse vocabulary and complex sentence structures, making it suitable for training language models. Here's a detailed overview of the data:

### Source of the Data

The Shakespearian text data includes various works of William Shakespeare, which are publicly available and free to use. These texts include plays, sonnets, and poems written by Shakespeare, known for their rich and intricate use of the English language.

### Data Files

The dataset is organized into several text files:
- **`qac_background.txt`**: Contains background information and context for the training data. This file includes introductory text and general information that provides a context for understanding the dataset.
- **`qac_training.tsv`**: The main training data file. This file contains the text that will be used to train the bigram and trigram models. The text is formatted as tab-separated values (TSV) for easy parsing.
- **`qac_validation.tsv`**: Contains validation data used to evaluate the performance of the model during training. This data is also formatted as TSV and includes excerpts from Shakespearian texts.

### Preprocessing

The raw Shakespearian text data undergoes several preprocessing steps to prepare it for training the QAC model:

1. **Lowercasing**: The text is converted to lowercase to maintain consistency and reduce the complexity of the character set.
2. **Special Tokens**: Special tokens `<s>` and `</s>` are added to mark the beginning and end of text sequences. These tokens help the model understand the boundaries of the sequences.
3. **Character Set Creation**: A unique set of characters is extracted from the dataset, including all the characters that appear in the text and the special tokens `<s>` and `</s>`. This set forms the basis for the bigram and trigram calculations.

### Usage of Data

The preprocessed data is then used to train the QAC model by calculating the frequencies of bigrams and trigrams. These frequencies are used to create a probability table, which helps in predicting the next character based on the given context.

### Significance of Shakespearian Data

Using Shakespearian text data provides several advantages:

- **Rich Vocabulary**: Shakespeare's works are known for their extensive and diverse vocabulary, which helps in creating a comprehensive language model.
- **Complex Structures**: The complex sentence structures in Shakespearian texts help in training models that can handle intricate language patterns.
- **Historical and Literary Value**: The texts have significant historical and literary value, making the dataset both interesting and challenging for language modeling tasks.

By leveraging Shakespearian data, the QAC model aims to provide accurate and contextually relevant auto-completion suggestions, enhancing the user experience in applications like text editors and search engines.

## Training 

The code and detailed usage instructions are provided in the [Training](training.ipynb) Jupyter Notebook.

## Generating Text

The code and detailed usage instructions are provided in the [Generation](Generating.ipynb) Jupyter Notebook.
