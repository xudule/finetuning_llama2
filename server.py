from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from inference import *

set_seed(42)

app = Flask(__name__)

model_name = "test_trainer"
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).half()
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = model.to("cuda")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['question']
    q_prompt = generate_prompt(question)

    answer = inference(q_prompt, model, tokenizer, max_output_length=1000)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    print("Server is stopping...")