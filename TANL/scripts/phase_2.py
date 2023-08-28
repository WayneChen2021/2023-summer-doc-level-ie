import json
import os
import argparse
from copy import deepcopy

def main(in_file, out_file):
    os.system("python3 process_raw_outputs.py --model_outs {} --output_file temp.json --tanl_file ../../data/multi_phase/tanl_train.json --gtt_file ../../data/multi_phase/gtt_train.json --debug_file process_raw_outputs_debug.json".format(in_file))

    with open("temp.json", "r") as f:
        infos = json.loads(f.read())
    os.remove("temp.json")
    
    tanl_formatted = []
    for document in infos.values():
        tanl_input = {
            "entities": [],
            "triggers": [],
            "relations": []
        }
        tanl_output = deepcopy(tanl_input)
        tanl_output["entities"] = document["entities"]
        tanl_output["triggers"] = document["triggers"]
        tanl_output["relations"] = document["relations"]

        if "pred_relations" in document:
            for i, template in enumerate(document["pred_relations"]):
                trigger = document["pred_triggers"][i]
                trigger = {
                    "type": trigger[0],
                    "start": trigger[1],
                    "end": trigger[2]
                }
                if not trigger in tanl_input["triggers"]:
                    tanl_input["triggers"].append(trigger)

                for relation in template:
                    entity = {
                        "type": "template entity",
                        "start": relation[1][0],
                        "end": relation[1][1]
                    }
                    if not entity in tanl_input["entities"]:
                        tanl_input["entities"].append(entity)
                    
                    tanl_input["relations"].append({
                        "type": relation[0],
                        "head": tanl_input["entities"].index(entity),
                        "tail": tanl_input["triggers"].index(trigger)
                    })
                    
        tanl_formatted.append({
            "tokens": document["tokens"],
            "input": tanl_input,
            "output": tanl_output
        })
    
    with open(out_file, "w") as f:
        f.write(json.dumps(tanl_formatted))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_train", type=str, required=True)
    parser.add_argument("--out_test", type=str, required=True)
    parser.add_argument("--out_train_short", type=str, required=True)
    parser.add_argument("--out_test_short", type=str, required=True)
    parser.add_argument("--train_samples", type=int, required=False, default=10)
    parser.add_argument("--test_samples", type=int, required=False, default=10)
    args = parser.parse_args()

    main(os.path.join(args.in_dir, "dev_predictions.txt"), args.out_train)
    with open(args.out_train, "r") as f:
        info = json.loads(f.read())
        with open(args.out_train_short, "w") as f2:
            f2.write(json.dumps(info[:args.train_samples]))

    main(os.path.join(args.in_dir, "test_predictions.txt"), args.out_test)
    with open(args.out_test, "r") as f:
        info = json.loads(f.read())
        with open(args.out_test_short, "w") as f2:
            f2.write(json.dumps(info[:args.test_samples]))