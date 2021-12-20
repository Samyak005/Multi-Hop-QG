import torch
import logging

from transformers import AutoTokenizer, AutoModelForPreTraining, AutoConfig

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CPU may not work, got to check.
device = torch.device('cpu') # CPU may not work, got to check.
print('Using device:' + str(device))
MODEL = 'gpt2'
SEQ_LENGTH = 600
MAXLEN = 768

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, #change
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,         #change                             
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()
    return model

tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)

gpt2_model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='./gpt_model_output-final/checkpoint-276/pytorch_model.bin')


def inference(answer, context, model, device):
    prompt = SPECIAL_TOKENS['bos_token'] + context + \
            SPECIAL_TOKENS['sep_token'] + answer + SPECIAL_TOKENS['sep_token']
            
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cpu")
    generated = generated.to(device)
    sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                min_length=3, 
                                max_length=MAXLEN,
                                top_k=30,                                 
                                top_p=0.7,        
                                temperature=0.9,
                                repetition_penalty=2.0,
                                num_return_sequences=1
                                )
    
    for i, sample_output in enumerate(sample_outputs):
        question = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(context) + len(answer)
        logging.debug("Decoded string" + question[a:] + "\n")    
        print("\n" + question[a:])
        return question[a:]
    
def get_inference2(answer, context):
    return inference(answer, context, gpt2_model, device)

if __name__ == "__main__":
    answer = "a fusional language"
    context = "Typologically, Estonian represents a transitional form from an agglutinating language to a fusional language. The canonical word order is SVO (subject–verb–object)."
    inference(answer, context, gpt2_model, device)
