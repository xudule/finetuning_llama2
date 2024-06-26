from inference import *
from dataset import *
import pandas as pd
import argparse

set_seed(42)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", help="model path", type=str, default="test_trainer", nargs='?')
args, unknown = parser.parse_known_args()

qa_dataset = read_and_prepocess_dataset()
max_output_length = find_max_length_of_output(qa_dataset)

model_dir = args.model
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).half()
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

start_time = time.time()
model = model.to("cuda")
predictions = []

exact = 0
for i in qa_dataset:
    is_match = False
    input = i['question']
    p = inference(input, model, tokenizer, max_output_length=max_output_length)
    if p.strip() == i['answer'].strip():
        is_match = True
        exact +=1
    predictions.append([i['question'], p, i['answer'], is_match])

print(f"Evaluation time: {time.time() - start_time} seconds")

df = pd.DataFrame(predictions, columns=["question", "predicted_answer", "target_answer", "is_exact_match"])

output_file = model_dir + '/evaluation.html'
df.to_html(output_file)

print(f'Number of exact match = {exact} (over {len(qa_dataset)})')
print("See details in ", output_file)