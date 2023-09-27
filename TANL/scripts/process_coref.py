import argparse
import os
import json
from process_og_muc import span_overlaps, build_entity
from nltk.tokenize import TreebankWordTokenizer as tbwt

def remove_overlap_spans(spans):
    while True:
        has_overlap = False
        new_spans = set()
        for span in spans:
            overlapping = [s for s in spans if span_overlaps(s, span)]
            new_spans.add((min(overlapping, key=lambda x:x[0])[0], max(overlapping, key=lambda x:x[1])[1]))
            if len(overlapping) > 1:
                has_overlap = True
        
        spans = new_spans
        if not has_overlap:
            break
    
    return spans

def main(gtt_info):
    examples = []
    for gtt_example in gtt_info:
        coref_groups = []
        token_spans = list(tbwt().span_tokenize(gtt_example["doctext"]))
        for template in gtt_example:
            for role, entities in template.items():
                if role != "incident_type":
                    for coref_set in entities:
                        coref_list = []
                        for span in coref_set:
                            head = gtt_example["doctext"].find(span)
                            assert head != -1

                            coref_list.append((head, head + len(span)))
                        coref_list = remove_overlap_spans(coref_list)
                        coref_list = sorted(coref_list, key = lambda tup : tup[0])
                        
                        new_coref_list = []
                        for span_tup in coref_list:
                            span_info = build_entity("", token_spans, span_tup[0], span_tup[1])
                            new_coref_list.append({"start": span_info["start"], "end": span_info["end"]})
                        coref_groups.append(new_coref_list)
        
        examples.append({
            "coref_groups": coref_groups,
            "tokens": [gtt_example["doctext"][tup[0] : tup[1]] for tup in token_spans],
            "id": gtt_example["docid"]
        })
    
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--muc_dir", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args

    os.system("python3 empty_test_examples.py --muc_dir {} --gtt_out gtt_out.json".format(
        args.muc_processing_file, args.muc_dir, args.annotation_file
    ))
    
    with open("gtt_out.json", "r") as f:
        gtt_info = json.loads(f.read())
    
    os.remove("gtt_out.json")