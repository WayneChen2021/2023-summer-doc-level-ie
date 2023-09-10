import argparse
import json
import numpy as np
import re
from itertools import product
from scipy.optimize import linear_sum_assignment

"""
in_file format:

[
    {
        "docid": "DEV-MUC3-0300",
        "tokens": ["NLP", "is", "awesome"]
        "pred_triggers": [(1, 2)],
        "pred_args": [(3, 4)],
        "gold_triggers": [[(1, 2), (3, 4)]],
        "gold_args": [[(5, 6), (7, 8)]],
    }
]
"""

def is_match(str1, str2, relax):
    if str1 == str2:
        return True

    if relax and (str1 in str2 or str2 in str1):
        return True

    return False


def count_matches(pred, gold, relax):
    larger = gold
    if len(pred) > len(gold):
        larger = pred

    cost_matr = np.zeros((len(larger), len(larger)))
    for i in range(len(larger)):
        for j in range(len(larger)):
            strings_match = i < len(pred) and j < len(gold)
            if strings_match:
                strings_match = is_match(pred[i], gold[j], relax)

            cost_matr[i][j] = -1 * int(strings_match)

    row_ind, col_ind = linear_sum_assignment(cost_matr)
    matches = int(-1 * cost_matr[row_ind, col_ind].sum())
    gold_matches = [gold_str for i, gold_str in enumerate(gold) if cost_matr[row_ind[i]][col_ind[i]] != 0]

    return matches, gold_matches

def non_zero(x, delta=1e-5):
    if x < delta:
        return 1
    
    return x

def argmax_f1(pred_trigs, pred_args, gold_trigs, gold_args, tokens, relax_match):
    regex = re.compile('[^a-zA-Z]')

    pred_trigs = [regex.sub("", "".join(tokens[tup[0] : tup[1]])) for tup in pred_trigs]
    gold_trigs = [regex.sub("", "".join(tokens[tup[0] : tup[1]])) for tup in gold_trigs]
    (num, gold_matches_trig), precision_den = count_matches(
        pred_trigs, gold_trigs, relax_match), len(pred_trigs)
    recall_den = len(gold_trigs)

    best_f1 = 0
    best_match_count, best_gold_args_len = 0, 0
    pred_args = [regex.sub("", "".join(tokens[tup[0] : tup[1]])) for tup in pred_args]
    for comb in product(*gold_args):
        comb_as_list = list(set(comb))
        match_count, gold_matches_ent = count_matches(pred_args, [regex.sub("", s) for s in comb_as_list], relax_match)

        sample_precision_num = num + match_count
        sample_precision_den = max(precision_den + len(pred_args), 1)
        sample_recall_num = sample_precision_num
        sample_recall_den = max(recall_den + len(comb_as_list), 1)
        sample_precision, sample_recall = sample_precision_num / \
            sample_precision_den, sample_recall_num / sample_recall_den
        sample_f1 = 2 * sample_precision * sample_recall / non_zero(sample_precision + sample_recall)

        if sample_f1 > best_f1:
            best_f1 = sample_f1
            best_match_count = match_count
            best_gold_args_len = len(comb_as_list)

    return num, precision_den, recall_den, best_match_count, len(pred_args), best_gold_args_len, gold_matches_trig, gold_matches_ent


def main(in_file, out_file, relax_match):
    with open(in_file, "r") as f:
        model_preds = json.loads(f.read())

    trig_num, trig_precision_den, trig_recall_den = 0, 0, 0
    ent_num, precision_den, recall_den = 0, 0, 0
    matches = {}
    for document in model_preds:
        # print(document)
        num_t, p_den_t, r_den_t, num, p_den, r_den, match_trig, match_ent = argmax_f1(
            document["pred_triggers"], document["pred_args"], document["gold_triggers"], document["gold_args"], document["tokens"], relax_match)
        try:
            assert num_t <= p_den_t and num_t <= r_den_t and num <= p_den and num <= r_den
        except Exception as e:
            print({
                "ent_num": ent_num,
                "p_den": p_den,
                "r_den": r_den,
                "trig_num": trig_num,
                "p_den_t": p_den_t,
                "r_den_t": r_den_t,
            })
            raise e

        matches[document["docid"]] = {
            "trig": match_trig,
            "ent": match_ent
        }

        ent_num += num
        precision_den += p_den
        recall_den += r_den
        trig_num += num_t
        trig_precision_den += p_den_t
        trig_recall_den += r_den_t        

    trig_precision = trig_num / max(trig_precision_den, 1)
    trig_recall = trig_num / max(trig_recall_den, 1)
    ent_precision = ent_num / max(precision_den, 1)
    ent_recall = ent_num / max(recall_den, 1)
    trig_f1 = 2 * trig_precision * trig_recall / non_zero(trig_precision + trig_recall)
    ent_f1 = 2 * ent_precision * ent_recall / non_zero(ent_precision + ent_recall)
    total_precision = (trig_num + ent_num) / max(trig_precision_den + precision_den, 1)
    total_recall = (trig_num + ent_num) / max(trig_recall_den + recall_den, 1)
    total_f1 = 2 * total_precision * total_recall / non_zero(total_precision + total_recall)

    info = json.dumps({
        "total": {
            "precision": total_precision,
            "recall": total_recall,
            "f1": total_f1
        },
        "trigger precision": trig_precision,
        "trigger recall": trig_recall,
        "trigger f1": trig_f1,
        "entity precision": ent_precision,
        "entity recall": ent_recall,
        "entity f1": ent_f1,
        "matches": matches
    }, indent=4)

    if out_file:
        with open(out_file, "w") as f:
            f.write(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, required=True)
    parser.add_argument("--o", type=str, required=True)
    parser.add_argument("--relax", action='store_true')
    args = parser.parse_args()

    main(args.i, args.o, args.relax)
