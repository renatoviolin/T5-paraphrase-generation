# %%
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# %%
model = T5ForConditionalGeneration.from_pretrained('./t5_paraphraser').eval()
tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# %%
sentence = "if I want to build a desktop software, what would be good languages to use?"
# sentence = "What are the ingredients required to bake a perfect cake?"
# sentence = "What is the best possible approach to learn aeronautical engineering?"
# sentence = "Do apples taste better than oranges in general?"

text = "paraphrase: " + sentence + " </s>"
encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
with torch.no_grad():
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=100,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10
    )


print(f"Original Question :: {sentence}")
print("\n")
print("==========Paraphrased Questions=========== ")
final_outputs = []
for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i, final_output))


# %%
