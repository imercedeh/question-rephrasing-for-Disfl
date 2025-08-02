# %pip install transformers datasets evaluate bert_score

root=".\\"

import sys
import os

sys.path.append(os.path.abspath(root))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq #BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset, DatasetDict
from datasets import Dataset
from evaluate import load
import torch.nn.functional as F
import numpy as np
import json
import torch

class DisfluencyCorrectorTester:
    def __init__(self, model_checkpoint,padding_len):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.padding_len=padding_len
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.bertscore = load("bertscore")
        self.bleu = load("bleu")
        self.exact_match = load("exact_match")

    def json_to_dataset(self,path):
        # Load the raw JSON
      with open(path, "r") as f:
            raw_data = json.load(f)

        # Convert to list of dicts
      data_list = [{"original": v["original"], "disfluent": v["disfluent"]} for v in raw_data.values()]

        # Create Hugging Face Dataset
      dataset = Dataset.from_list(data_list)
      return dataset

    def make_predictions(self,disfluent_sentences):
      from torch.utils.data import DataLoader

      batch_size = 8
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model.to(device)

      dataloader = DataLoader(disfluent_sentences, batch_size=batch_size)
      predictions = []

      self.model.eval()
      for batch in dataloader:
          inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.padding_len)
          input_ids = inputs["input_ids"].to(device)
          attention_mask = inputs["attention_mask"].to(device)

          with torch.no_grad():
              outputs = self.model.generate(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  max_length=self.padding_len,
                  num_beams=4,
                  early_stopping=True
              )

          decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
          predictions.extend(decoded)
      return predictions

    def evaluate_with_bertscore(self,predictions,references):
        bertscore_res = self.bertscore.compute(predictions=predictions, references=references, lang="en")
        R=np.mean(bertscore_res['recall'])
        P=np.mean(bertscore_res['precision'])
        F1=np.mean(bertscore_res['f1'])
        self.eval_results["BERTScore Recall"]=R
        self.eval_results["BERTScore Precision"]=P
        self.eval_results["BERTScore F1"]=F1

        print("BERTScore Results:")
        print(f"Precision: {P:.4f}")
        print(f"Recall:    {R:.4f}")
        print(f"F1:        {F1:.4f}",'\n')

    def evaluate_with_bleu(self,predictions,references):
        bleu_res = self.bleu.compute(predictions=predictions, references=references)
        self.eval_results["BLEU"]=bleu_res["bleu"]

        print("BLEU Score:")
        print(bleu_res["bleu"],'\n')

    def evaluate_with_exact_match(self,predictions,references):
        results = self.exact_match.compute(predictions=predictions, references=references)
        self.eval_results["exact_match"]=results["exact_match"]

        print("Exact Match Results:")
        print(round(results["exact_match"], 4),'\n')

    def evaluate(self,path,res_path):
      self.eval_results={}
      print("Loading dataset from the file.....................................................")
      dataset=self.json_to_dataset(path)
      targets= dataset["original"]
      input_sentences = dataset["disfluent"]

      print("Generating predictions............................................................")
      predictions = self.make_predictions(input_sentences)

      print("Calculating scores................................................................\n")
      self.evaluate_with_bleu(predictions=predictions,references=targets)
      self.evaluate_with_bertscore(predictions=predictions,references=targets)
      self.evaluate_with_exact_match(predictions=predictions,references=targets)

      with open(res_path+".json", "w") as f:
        json.dump(self.eval_results, f, indent=4)

train_path=root+"Dataset\\train.json"
dev_path=root+"Dataset\\dev.json"
test_path=root+"Dataset\\test.json"

results_path=root+"Models\\BART\\Results\\"

model_checkpoint = root+"Models/BART"#"facebook/bart-base"

test_model=DisfluencyCorrectorTester(model_checkpoint,32)

test_model.evaluate(train_path,results_path+"mybart-train")

test_model.evaluate(dev_path,results_path+"mybart-dev")

test_model.evaluate(test_path,results_path+"mybart-test")

