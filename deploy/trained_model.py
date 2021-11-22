
import torch
import logging
# Transformer version 4.9.1 - Newer versions may not work.
from transformers import AutoTokenizer

def t5_supp_inference(review_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU may not work, got to check.
    # device = torch.device('cpu')
    print('Using device:' + str(device))
    PRETRAINED_MODEL = 't5-base'
    SEQ_LENGTH = 600
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )

    model = torch.load("../trained_models/t5_model_hotpot_supporting_facts_last.pth")
    model.eval()
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
    # device.empty_cache()
    del model
    del tokenizer
    return decoded_string

def t5_full_inference(review_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU may not work, got to check.
    # device = torch.device('cpu')
    print('Using device:' + str(device))
    PRETRAINED_MODEL = 't5-base'
    SEQ_LENGTH = 600
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )

    model = torch.load("../trained_models/t5_model_hotpot_full_context_last.pth")
    model.eval()
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
    # device.empty_cache()
    del model
    del tokenizer
    return decoded_string

def bart_supp_inference(review_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU may not work, got to check.
    # device = torch.device('cpu')
    print('Using device:' + str(device))
    PRETRAINED_MODEL = 'facebook/bart-base'
    SEQ_LENGTH = 600
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )

    model = torch.load("../trained_models/bart_model_hotpot_supporting_facts_last.pth")
    model.eval()
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
    # device.empty_cache()
    del model
    del tokenizer
    return decoded_string

def bart_full_inference(review_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU may not work, got to check.
    # device = torch.device('cpu')
    print('Using device:' + str(device))
    PRETRAINED_MODEL = 'facebook/bart-base'
    SEQ_LENGTH = 600
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )

    model = torch.load("../trained_models/bart_model_hotpot_full_context_last.pth")
    model.eval()
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
    # device.empty_cache()
    del model
    del tokenizer
    return decoded_string

# if __name__ == "__main__":
#     review_text = "<answer> a fusional language <context> Typologically, Estonian represents a transitional form from an agglutinating language to a fusional language. The canonical word order is SVO (subject–verb–object)."
#     t5_supp_inference(review_text, md2, device)


def get_inference(answer, context, model_name):
    valuation_text = "<answer> " + answer + " <context> " + context

    if model_name == 't5_supp':
        return t5_supp_inference(valuation_text)
    elif model_name == 't5_full':
        return t5_full_inference(valuation_text)
    elif model_name == 'bart_supp':
        return bart_supp_inference(valuation_text)
    elif model_name == 'bart_full':
        return bart_full_inference(valuation_text)
