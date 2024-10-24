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

The dataset used in this project is the [IMDb movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/). It contains 50,000 reviews categorized as positive or negative. These reviews are split into 25,000 training examples and 25,000 testing examples.

- **Training set:** 25,000 movie reviews (50% positive, 50% negative)
- **Test set:** 25,000 movie reviews (50% positive, 50% negative)

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

**Key Dependencies:**
- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `numpy`
- `pandas`

### Key Hyperparameters:
- **Batch Size:** 16
- **Learning Rate:** 5e-5
- **Epochs:** 3

You can modify these parameters in the `config.yaml` file.

## Results

Here are the results of the fine-tuned DistilBERT model on the IMDb test set:

- **Accuracy:** XX%
- **F1 Score:** XX%
- **Precision:** XX%
- **Recall:** XX%

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
