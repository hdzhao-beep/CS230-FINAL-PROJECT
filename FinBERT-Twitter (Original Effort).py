from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments, Trainer, evaluate
import pickle

# Load the preprocessed variables from the pickle file
with open('preprocessed_data.pickle', 'rb') as f:
    X, y, X_test, y_test, df_train, df_test = pickle.load(f)

class FIN_twitterSentimentClassifier():
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_precision = evaluate.load("precision")
        self.metric_recall = evaluate.load("recall")   
        
    def tokenize_function(self, batch):
        # Tokenize the input batch
        tokenized_batch = self.tokenizer(batch["text"], padding='max_length', truncation=True, max_length=125)
        tokenized_batch["label"] = batch["label"]
        return tokenized_batch
    
    def import_tweets(self, data):
        # Split the data into train and test sets
        self.train, self.test = train_test_split(data, train_size=200000, test_size=5000, random_state=2018012198)
        
        # Convert train and test sets to Datasets
        self.train = Dataset.from_pandas(self.train)
        self.test = Dataset.from_pandas(self.test)
        
        # Apply tokenization to train and test datasets
        self.train_dataset = self.train.map(self.tokenize_function, batched=True)
        self.test_dataset = self.test.map(self.tokenize_function, batched=True)
        
    def get_metrics(self, predictions, labels):
        # Compute evaluation metrics
        accuracy = self.metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        precision = self.metric_precision.compute(predictions=predictions, references=labels)["precision"]
        recall = self.metric_recall.compute(predictions=predictions, references=labels)["recall"]
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1_score}
        
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return self.get_metrics(predictions, labels)

    def train_model(self):
        # Load the pre-trained model for sequence classification
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
        
        # Set up the training arguments
        self.training_args = TrainingArguments(
            output_dir="training_arguments", 
            evaluation_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=3e-5,
            weight_decay=0.01,
            num_train_epochs=3,
        )
        
        # Initialize the trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        self.trainer.train()
        
        # Move the model to CPU
        self.model = self.model.to("cpu")
        
    def save_model_local(self, path):
        # Save the model locally
        self.trainer.save_model(path)
            
    def load_saved_model(self, path):
        # Load the saved model
        self.model = DistilBertForSequenceClassification.from_pretrained(path, num_labels=self.num_labels)
        self.model = self.model.to("cpu")
        self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer)
        
    def get_scores(self, text):
        # Get the sentiment scores for the given text
        return self.pipe(text)
    
    def predict(self, text):
        label2id = {"LABEL_0": 0, "LABEL_1": 1}
        scores = self.get_scores(text)
        predicted = [label2id[sample["label"]] for sample in scores]
        return predicted
    
    def evaluate_model(self, df_test):        
        self.eval = Dataset.from_pandas(df_test)
        self.eval_dataset = self.eval.map(self.tokenize_function, batched=True) 
        predictions = np.array(self.predict(self.eval_dataset["text"]))
        labels = np.array(self.eval_dataset["label"])
        return self.get_metrics(predictions, labels)       

# Create an instance of the FIN_twitterSentimentClassifier
ts_new = FIN_twitterSentimentClassifier()

# Import the preprocessed tweets (df_train)
ts_new.import_tweets(df_train)

# Train the model
ts_new.train_model()

# Save the trained model locally
ts_new.save_model_local("new_model")

# Load the saved model
ts_new.load_saved_model("new_model")

# Evaluate the model on the test dataset (df_test)
ts_new.evaluate_model(df_test)
