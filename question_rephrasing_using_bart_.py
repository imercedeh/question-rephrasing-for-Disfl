# %pip install transformers datasets evaluate

root=".\\"

import sys
import os

sys.path.append(os.path.abspath(root))

"""# Defining the Model"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq #BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset, DatasetDict
from datasets import Dataset
from evaluate import load
import torch.nn.functional as F
import numpy as np
import json
import torch

class DisfluencyCorrector:
    def __init__(self, model_checkpoint, model_saving_path):
        self.model_checkpoint = model_checkpoint
        self.model_saving_path = model_saving_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.padding_len=32
        self.bertscore = load("bertscore")
        self.bleu = load("bleu")
        self.exact_match = load("exact_match")

    def json_to_dataset(self,path):
        # Load the raw JSON
      with open(path, "r") as f:
            raw_data = json.load(f)

        # Convert to list of dicts
      data_list = [{"original": v["original"], "disfluent": v["disfluent"]} for v in raw_data.values()]

      inputs_lengths  = [len(self.tokenizer.encode(item['disfluent'], add_special_tokens=False)) for item in data_list]
      targets_lengths = [len(self.tokenizer.encode(item['original'], add_special_tokens=False)) for item in data_list]

      print(f"Average input tokenized length: {sum(inputs_lengths) / len(inputs_lengths):.2f}")
      print(f"Average target tokenized length: {sum(targets_lengths) / len(targets_lengths):.2f}\n")

        # Create Hugging Face Dataset
      dataset = Dataset.from_list(data_list)
      return dataset

    # Preprocessing function
    # Tokenization
    def preprocess(self,sample_batch):
        inputs=[disfluent for disfluent in sample_batch["disfluent"]]
        targets=[original for original in sample_batch["original"]]

        model_input = self.tokenizer(inputs, truncation=True, padding="max_length", max_length=self.padding_len)

        with self.tokenizer.as_target_tokenizer():
              labels = self.tokenizer(targets, truncation=True, padding="max_length", max_length=self.padding_len)

        model_input["labels"] = labels["input_ids"]
        return model_input

    def get_tokenized_dataset(self,dataset_path):
      dataset=self.json_to_dataset(dataset_path)
      tokenized_dataset = dataset.map(self.preprocess, batched=True)
      return tokenized_dataset

    def compute_metrics(self,eval_preds):

        logits=eval_preds.predictions
        logits=logits[0]
        labels=eval_preds.label_ids

        if isinstance(logits, np.ndarray):
          logits = torch.tensor(logits)

        probabilities = F.softmax(logits, dim=-1)
        preds = torch.argmax(probabilities, dim=-1)

        decoded_preds = []
        decoded_labels = []

        # Decode predictions and references
        #Do batch processing to avoid memory out of range
        for i in range(0, len(preds), 32):
          batch_preds = preds[i:i+32]
          batch_labels = labels[i:i+32]

          # Convert -100 to pad_token_id so tokenizer can decode
          batch_labels = [[(l if l != -100 else self.tokenizer.pad_token_id) for l in label] for label in batch_labels]

          decoded_preds += self.tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
          decoded_labels += self.tokenizer.batch_decode(batch_labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        bleu_score = self.bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bertscore_res = self.bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        EM_score = self.exact_match.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "BLEU": bleu_score['bleu'],
            'Bertscore Recall':np.mean(bertscore_res['recall']),'Bertscore Precision':np.mean(bertscore_res['precision']),'Bertscore F1':np.mean(bertscore_res['f1']),
            'Exact Match':EM_score['exact_match']
        }

    def train_model(self,dataset_path,training_arguments):
      print("Loading datasets from files......................................................")
      self.tokenized_train = self.get_tokenized_dataset(dataset_path['train'])
      self.tokenized_dev = self.get_tokenized_dataset(dataset_path['dev'])

      self.training_args=training_arguments
      self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

      self.trainer = Trainer(
          model=self.model,
          args=training_args,
          train_dataset=self.tokenized_train,
          eval_dataset=self.tokenized_dev,
          tokenizer=self.tokenizer,
          data_collator=self.data_collator,
          compute_metrics=self.compute_metrics
      )

      print("Training started..................................................................")
      self.trainer.train()
      if self.model_saving_path!="Don't Save":
        self.trainer.save_model(self.model_saving_path)
        self.tokenizer.save_pretrained(self.model_saving_path)

"""# Running the code"""

train_path=root+"Dataset\\train.json"
dev_path=root+"Dataset\\dev.json"
test_path=root+"Dataset\\test.json"

model_checkpoint = "facebook/bart-base"
model_saving_path=root+"Models\\BART"

my_model=DisfluencyCorrector(model_checkpoint,model_saving_path=model_saving_path)

training_args = Seq2SeqTrainingArguments(
    output_dir=".\\",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=1,
    overwrite_output_dir=True,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_max_length=32,
    generation_num_beams=5,
    logging_dir='.\\',
    logging_strategy="epoch",
    #logging_steps=10,
    report_to="none",
    fp16=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
)

my_model.train_model(dataset_path={'train':train_path,'dev':dev_path},training_arguments=training_args)