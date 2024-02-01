from datasets import Dataset
import json

def read_dataset():
    with open('data.json', 'r') as f:
        qa_pairs = json.load(f)
    return qa_pairs

def generate_prompt(question):
    return "### Question:\n" + question + "\n### Answer:"

def read_and_tockenize_dataset(tokenizer):
    qa_pairs = read_dataset()
    # Combine the questions and answers into sequences
    sequences = [pair["question"] + pair["answer"] for pair in qa_pairs]
    tokenized_sequences = tokenizer(sequences, return_tensors="np", truncation=True, padding=True)
    labels = tokenized_sequences["input_ids"].copy()

    qa_dataset = Dataset.from_dict({
        'input_ids': tokenized_sequences["input_ids"],
        'attention_mask': tokenized_sequences["attention_mask"],
        'labels': labels,
    })

    return qa_dataset

def split_train_test(qa_dataset):
    return qa_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
