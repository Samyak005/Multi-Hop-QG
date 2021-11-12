import pandas as pd
from nlgeval import NLGEval
from collections import OrderedDict
import json

if __name__ == '__main__':
    # df = pd.read_csv('predicted_supporting_facts1.csv') # T5 Supporting Facts
    # df = pd.read_csv('predicted_full_context.csv') # T5 Full Context
    df = pd.read_csv('predicted_gpt2_context.csv') # GPT2 Full context (first 2)
    ref = df["question"].tolist()
    hyp = df["predicted"].tolist()

    n = NLGEval(metrics_to_omit=["CIDEr", "SkipThoughtCS",
                "EmbeddingAverageCosineSimilarity", "VectorExtremaCosineSimilarity", "GreedyMatchingScore"])

    scores = n.compute_metrics(ref_list=[ref], hyp_list=hyp)

    scores['Bleu_mean_eq_weight'] = (scores['Bleu_1'] + \
        scores['Bleu_2'] + scores['Bleu_3'] + scores['Bleu_4']) / 4.0

    scores = dict(OrderedDict(sorted(scores.items())))

    print(scores)
    with open('evaluation_results.json', 'w') as fp:
        json.dump(scores, fp, indent=4)
