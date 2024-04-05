# Finetuning llama2

This repository helps you finetune Llama2 with your dataset.

You will need to download Llama2, for example, see instructions from HuggingFace: https://huggingface.co/meta-llama

Adapt data.json with your database.

Go to *src* folder:
- Run *train.py* to finetune Llama2 with your dataset.
- Run *inference.py* to compare result from the original model and the finetuned one.

Run the finetuned model in chat mode: see src/chatbot/README.md