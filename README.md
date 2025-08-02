# Question Rephrasing with Transformer Models on Disfl-QA

This repository contains code for a question rephrasing task using transformer-based architectures (`T5-small` and `BART-base`) fine-tuned on the [Disfl-QA dataset](https://github.com/google-research-datasets/Disfl-QA). The goal is to convert disfluent or noisy user questions into clear, fluent versions.

---

## Repository Structure
├── Dataset/ # Dataset files
├── Models/T5/Results/ # T5 model Evaluation results
├── Models/BART/Results/ # BART model Evaluation results
├── test_t5_question_rephrasing.py # T5 testing script
├── question_rephrasing_using_t5_.py # Fine-tuning script for T5
├── question_rephrasing_using_bart_.py # Fine-tuning script for BART
├── test_bart_question_rephrasing.py # BART testing script
└── README.md # This file

## Dependencies

This project requires the following Python libraries:

- `transformers` 
- `datasets`
- `torch` (PyTorch)
- `numpy`
- `scikit-learn`
- `bert-score`
- `evaluate`
