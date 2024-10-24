# Fine-Tuning DistilBERT on IMDb Movie Reviews

This repository contains code for fine-tuning the DistilBERT model on the IMDb dataset to classify movie reviews as either positive or negative. 
After fine-tuning, the model is used for inference on new movie reviews.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)

## Project Overview

The goal of this project is to fine-tune [DistilBERT](https://arxiv.org/abs/1910.01108), a smaller and faster version of BERT, on the IMDb dataset.<br>
DistilBERT is well-suited for tasks like sentiment analysis, where we aim to determine whether a movie review expresses a positive or negative sentiment.<br>

## Dataset

The dataset used in this project is the [IMDb movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/). It contains 100,000 reviews categorized as positive, negative and unlabelled.

## Model

I'm fine-tuning [DistilBERT](https://huggingface.co/distilbert-base-uncased), a lightweight version of BERT, using the Hugging Face `transformers` library. The model is initialized with pre-trained weights and further trained on the IMDb dataset for binary sentiment classification.

### Key Features of DistilBERT:
- 60% faster than BERT
- Retains 97% of BERT’s language understanding capabilities
- Smaller in size, making it more efficient for inference

## Requirements

Before getting started, install the necessary libraries by running the following command:

```bash
pip install -r requirements.txt
```

```bash
pip install transformers datasets evaluate accelerate
```
**Key Dependencies:**
- `transformers`
- `datasets`
- `evaluate`
- `accelerate`
- `torch`
- `numpy`

### Key Hyperparameters:
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Epochs:** 2

## Results

Here are the results of the fine-tuned DistilBERT model on the IMDb test set:

- **Accuracy:** XX%
- **F1 Score:** XX%
- **Precision:** XX%
- **Recall:** XX%

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
