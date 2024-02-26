from inference import *
from dataset import *
import pandas as pd

set_seed(42)

qa_dataset = read_dataset()

model_dir = "test_trainer"
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).half()
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

start_time = time.time()
model.to("cuda")
predictions = []

exact = 0
for i in qa_dataset:
    is_match = False
    input = i['question']
    p = inference(input, model, tokenizer)
    if p.strip() == i['answer'].strip():
        is_match = True
        exact +=1
    predictions.append([i['question'], p, i['answer'], is_match])

df = pd.DataFrame(predictions, columns=["question", "predicted_answer", "target_answer", "is_exact_match"])

output_file = 'evaluation.html'
df.to_html(output_file)

print(f'Number of exact match = {exact} (over {len(qa_dataset)})')
print("See details in ", output_file)