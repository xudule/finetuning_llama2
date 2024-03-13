from datasets import Dataset
import json

def read_dataset(file_name='data.json'):
    with open(file_name, 'r') as f:
        ds = json.load(f)
    return ds

def generate_prompt(question):
    return "### Question:\n" + question + "\n### Answer:"

def read_and_prepocess_dataset():
    ds = read_dataset()
    ds = [{'question': generate_prompt(item['question']), 'answer': item['answer']} for item in ds]
    return ds

def find_max_length_of_output(qa_dataset_with_prompt):
    return max([len(item['question']) + len(item['answer']) for item in qa_dataset_with_prompt])

def read_and_tockenize_dataset(tokenizer):
    qa_pairs = read_and_prepocess_dataset()
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

if __name__ == "__main__":
    ds = read_dataset()
    print(ds[0])
    ds = read_and_prepocess_dataset()
    print(ds[0])
    print("Max length of output for model should be: ", find_max_length_of_output(ds))
