
import torch
import logging
# Transformer version 4.9.1 - Newer versions may not work.
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU may not work, got to check.
print('Using device:' + str(device))
PRETRAINED_MODEL = 't5-base'
SEQ_LENGTH = 600
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

tokenizer.add_special_tokens(
    {'additional_special_tokens': ['<answer>', '<context>']}
)

md2 = torch.load("model_hotpot_last.pth")


def inference(review_text, model, device):
    encoded_text = tokenizer(
        review_text,
        padding=True,
        max_length=SEQ_LENGTH,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    input_ids = encoded_text['input_ids']
    with torch.no_grad():
        output = model.generate(input_ids)
    decoded_string = tokenizer.decode(output[0], skip_special_tokens=True)
    logging.debug("Decoded string" + decoded_string)
    print(decoded_string)

    return decoded_string


if __name__ == "__main__":
    review_text = "<answer> a fusional language <context> Typologically, Estonian represents a transitional form from an agglutinating language to a fusional language. The canonical word order is SVO (subject–verb–object)."
    inference(review_text, md2, device)


def get_inference(answer, context):
    valuation_text = "<answer> " + answer + " <context> " + context
    return inference(valuation_text, md2, device)
