# Import Libraries

import numpy as np
import torch
import evaluate
import warnings
warnings.filterwarnings("ignore")

# hugging face libs
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm

from torch.optim import AdamW
from torch.utils.data import DataLoader

###### - Load Dataset - ######
imdb = load_dataset("imdb")

# example text with label
imdb["test"][0]

###### - Preprocessing - ######

# Tokenize text field by using a DistilBERT tokensizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Create a preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors = "pt")

tokenized_imdb = imdb.map(preprocess_function, batched=True)

# Build tokens by padding them to match length of maximum sentence length of an item in IMDB dataset
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

torch.cuda.empty_cache()

# Remove text column from the imdb dataset
tokenized_imdb = tokenized_imdb.remove_columns(["text"])

# Rename the label column to "labels"
tokenized_imdb = tokenized_imdb.rename_column("label", "labels")

# Convert lists to pyTorch tensors
tokenized_imdb.set_format("torch")

# Create smaller subsets of data to speed up fine tuning
small_train_dataset = tokenized_imdb["train"].shuffle(seed=42).select(range(1000))

###### -  Compute Metrics - ######

# load accuracy metric 
accuracy = evaluate.load("accuracy")

# create a function that computes metrics to be pushed into the training loop (while re-training DistilBERT)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Data Loader for creating batches

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# Since we have labelled text (either Positive or Negative), we need to  map the ID to Label and Label to ID in our model as well
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# train model by loading distilbert model, labels and mappings
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Optimizer 

#Create an optimizer and learning rate scheduler to fine-tune the model. For this dataset, I'm using AdamW optimizer from PyTorch
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning Rate

# Get the default learning rate scheduler from Transfromers lib
num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

###### -  Training Begins - ######

progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluate

metric = evaluate.load("accuracy")

model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()





