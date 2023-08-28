import re
import json
import ast
import argparse
from collections import defaultdict

#CREDITS GO TO BARRY FOR COMING UP WITH MOST OF THIS SCRIPT

trigger_predict_prefix = "trigger_output_sentence"  # Printing of this indicates a start of a new example
gold_template_prefix = " gt_relations"  # Note there is a space. Could be multiple per example. Ignorable.
pred_template_prefix = "predicted_relations"  # Could be multiple per example
triggers_prefix = " triggers "

trigger_sample = "trigger_output_sentence today , medellin , colombia 's second largest city , once again experienced " \
                 "a terrorist escalation when seven bank branch offices were shaken by explosives that caused heavy " \
                 "damage but no fatalities , according to radio reports broadcast in bogota ( 500 km to the south ) . " \
                 "the targets of the [ attacks | attack ] were the banco cafetero branches and its offices in " \
                 "medellin 's middle , western , and southeastern areas . according to preliminary reports , " \
                 "over 55 kg of [ dynamite | bombing ] were used in the attacks . the radio report added that the " \
                 "police defused another 20 kg of explosives that had slow burning fuses . the medellin cartel " \
                 "operates in this city located in colombia 's northeastern area . for several days now , " \
                 "the city has been shaken by army and police operations in an unprecedented action to capture drug " \
                 "lords . no one has claimed responsibility for the terrorist attacks , which lasted for 1 hour ."


def letters(input):
    valids = []
    for character in input:
        if character.isalpha() or character == " ":
            valids.append(character)
    return ''.join(valids)


def extract_trigger(s, org_text):
    tokens = org_text
    regex_str = r"\[[a-z0-9\-.,' ]* \| [a-z\-. }{]*\]" # COMMENT: may need to change for edge case, different TANL schema
    splits_re = re.split(regex_str, s)
    triggers_re = re.findall(regex_str, s)
    # index = 0
    if len(triggers_re) == 0:
        return []
    triggers = []
    assert len(splits_re) - 1 == len(triggers_re)
    for i, no_trigger_part in enumerate(splits_re):
        if i == len(triggers_re):
            break
        trigger = triggers_re[i].split("|")[0][1:].strip()
        inc_type = triggers_re[i].split("|")[1][:-1].strip()
        triggers.append((trigger, " ".join(inc_type.split(" ")[2:-1])))
    return triggers

# COMMENT: may need to change depending on model output
role_map = {
    "perpetrating individual": "PerpInd",
    "perpetrating organization": "PerpOrg",
    "target argument": "Target",
    "victim argument": "Victim",
    "weapon argument": "Weapon"
}

def run_permutation(doc_index, output_path, muc_tanl_input_path, muc_gtt_input_path, save_path, provided_triggers, print_lines, debug_file, derive_start=False):
    id_to_templates = defaultdict(lambda: defaultdict(list))

    f = open(output_path)
    lines = f.readlines()
    f.close()

    f = open(muc_tanl_input_path)
    tanl_input_file = json.load(f)
    f.close()

    f = open(muc_gtt_input_path)
    gtt_input_file = json.load(f)
    f.close()

    num_of_examples = len(tanl_input_file)
    
    start_from = 0
    cur_triggers = []
    for count_ind, line in enumerate(lines[start_from:], start_from):
        # if print_lines:
        #     print(line, count_ind)
        if line.startswith(triggers_prefix):
            if "set()" in line:
                cur_triggers = []
            else:
                cur_triggers = list(ast.literal_eval(line[len(triggers_prefix):]))
            id_to_templates[doc_index]['pred_triggers'] = cur_triggers

        if line[:len(trigger_predict_prefix)] == trigger_predict_prefix:

            # Found predicted text with marked triggers, potentially none
            # This means a new example for sure. incrementing doc_index.
            doc_index += 1
            if doc_index < 0:
                continue
            if doc_index >= num_of_examples:  # MUC test only has 200 examples.
                break
            assert 'raw_output' not in id_to_templates[doc_index]
            # Confirmed to process this example.
            cur_tokens = tanl_input_file[doc_index]['tokens']

            id_to_templates[doc_index]['pred_trigger_in_text'] = line[len(trigger_predict_prefix) + 1:]
            # found_match = False
            # for section, trigger_lst in provided_triggers.items():
            #     if section in line:
            #         cur_triggers = trigger_lst
            #         found_match = True
            #         break
            # if not found_match:
            #     cur_triggers = extract_trigger(line[len(trigger_predict_prefix) + 1:], cur_tokens)
            id_to_templates[doc_index]['gold_templates'] = gtt_input_file[doc_index]['templates']
            id_to_templates[doc_index]['doctext'] = gtt_input_file[doc_index]['doctext']
            # diff_score = len(set(nltk.casual_tokenize(id_to_templates[doc_index]['pred_trigger_in_text'])).difference(
            #     nltk.casual_tokenize(id_to_templates[doc_index]['doctext'])))
            # assert diff_score < 20, diff_score  # two text should be extremely similar to each other.
            for k, v in tanl_input_file[doc_index].items():
                id_to_templates[doc_index][k] = v
            template_id = -1

        if doc_index < 0:
            continue
        if doc_index >= num_of_examples:  # MUC test only has 200 examples.
            break

        id_to_templates[doc_index]['raw_output'].append(line)

        empty_template = {
            "incident_type": None,
            "PerpInd": [],
            "PerpOrg": [],
            "Target": [],
            "Victim": [],
            "Weapon": []
        }
        if line[:len(pred_template_prefix)] == pred_template_prefix:
            # Found one template, potentially with no role fillers (`set()`).
            template_id += 1
            template_str = line[len(pred_template_prefix):]
            template = empty_template
            if "set()" in template_str:
                # continue
                # no role fillers
                if len(cur_triggers):
                    template['incident_type'] = cur_triggers[template_id][1]
                id_to_templates[doc_index]['pred_relations'].append([])
            else:
                # at least one role fillers predicted for this template
                # print(template_str)
                parsed = ast.literal_eval("[" + template_str[template_str.index("{") + 1:template_str.rindex(
                    "}")] + "]")  # Assuming each line is of form " {(..)}\n"
                id_to_templates[doc_index]['pred_relations'].append(parsed)
                for argument in parsed:
                    bracket_removed = argument[0].replace("{ ", "").replace(" }", "")
                    role = bracket_removed.split(":")[0].strip()
                    role = role_map[" ".join(role.split(" ")[:2])] # Comment out for multiple_triggers, else keep
                    template['incident_type'] = bracket_removed.split(" ")[-2].strip()
                    template[role].append([" ".join(cur_tokens[argument[1][0]:argument[1][1]])])

            id_to_templates[doc_index]['pred_templates'].append(template)

    result_dict = dict()
    result_dict_non_empty = dict()
    for index, value in id_to_templates.items():
        result_dict[index] = dict(value)
        if "pred_templates" not in result_dict[index]:
            result_dict[index]['pred_templates'] = []
        try:
            assert len(result_dict[index]["pred_templates"]) == len(value['pred_triggers'])
        except Exception:
            with open(debug_file, "w") as f:
                f.write(json.dumps(value))
            raise Exception
        result_dict[index] = {key: value for key, value in sorted(result_dict[index].items())}  # Sort
        if len(value['pred_triggers']) > 0:
            result_dict_non_empty[index] = result_dict[index]
            assert "pred_templates" in result_dict[index]

    json.dump(result_dict, open(save_path, "w+"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outs", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--tanl_file", type=str, required=True)
    parser.add_argument("--gtt_file", type=str, required=True)
    parser.add_argument("--edge_cases", type=str, required=False)
    parser.add_argument("--debug_file", type=str, required=True)
    parser.add_argument("--print_lines", type=bool, required=False, default=True)
    args = parser.parse_args()

    provided_triggers = {}
    if args.edge_cases:
        with open(args.edge_cases, "r") as f:
            provided_triggers = json.loads(f.read())

    run_permutation(-1, args.model_outs, args.tanl_file, args.gtt_file, args.output_file, provided_triggers, args.print_lines, args.debug_file, False)