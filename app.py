import flask
from flask import Flask, request, render_template
import json
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').eval()
tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

app = Flask(__name__)


def _generate(sentence, num_sentences, max_len, top_p, early_stop):
    text = "paraphrase: " + sentence + " </s>"
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    with torch.no_grad():
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=max_len,
            top_k=100,
            top_p=top_p,
            early_stopping=True if early_stop else False,
            num_return_sequences=num_sentences
        )
    final_outputs = []
    sentence = re.sub(r'[^\w\s]','',sentence)
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_paraphrase', methods=['post'])
def get_paraphrase():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        num_sentences = int(request.json['num_sentences'])
        max_len = int(request.json['max_len'])
        top_p = float(request.json['top_p'])
        early_stop = int(request.json['early_stop'])

        response = _generate(input_text, num_sentences, max_len, top_p, early_stop)
        str_response = '\n'.join([r for r in response])

        return app.response_class(response=json.dumps(str_response), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=True)
