from django.shortcuts import render
from django.http import HttpResponse

import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from huggingface_hub import notebook_login
notebook_login()

# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("C:/dclv/nlp_extend/django_web/web_demo/model_Transformer5_SummaryText")

# device = torch.device('cpu')

# def model_t5(text):
#     text = text

#     preprocess_text = text.strip().replace("\n", "")
#     t5_prepared_text = "summarize: " + preprocess_text

#     tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors = "pt").to(device)

#     summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)

#     summary_text = tokenizer.decode(summary_ids[0], skip_special_token=True)

#     return summary_text

def summarization_t5(sentence):
    text =  "vietnews: " + sentence + " </s>"
    encoding = tokenizer(text, return_tensors="pt")
    # input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        early_stopping=True
    )
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return line


# Create your views here.
def index(request):
    return render(request, "pages/home.html")

def summary(request):
    return render(request, 'pages/home.html')

def task_summary(request):
    if request.method == 'POST':
        text_input = request.POST['input_text']
        # text_summary = model_t5(text_input)
        text_summary = summarization_t5(text_input)
        return render(request, 'pages/home.html', {'text_input': text_input,'result': text_summary})
    return render(request, 'pages/home.html')