from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import argparse

import sys
sys.path.append('../')
from inference import *

set_seed(42)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", help="model path", type=str, default="../low_memory_mode", nargs='?')
args, unknown = parser.parse_known_args()

app = Flask(__name__)

model_name = args.model_name
print("Using model: ", model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).half()
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = model.to("cuda")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['question']
    q_prompt = generate_prompt(question)

    answer = inference(q_prompt, model, tokenizer, max_output_length=2000)
    print(f"Question: {question}, Answer: {answer}")
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    print("Server is stopping...")