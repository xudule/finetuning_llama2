# Finetuning llama2

This repository helps you finetune Llama2 with your dataset.

You will need to download Llama2, for example, see instructions from HuggingFace: https://huggingface.co/meta-llama

Create your database with json format and name it *data.json*. A database example can be seen in *data_template.json*
Add moderation in *data_moderation.json* to moderate your model. An example can be seen in *data_moderation_template.json*

Go to *src* folder:
- Run *train.py* to finetune Llama2 with your dataset.
- Run *inference.py* to compare result from the original model and the finetuned one.

Run the finetuned model in chat mode: see src/chatbot/README.md