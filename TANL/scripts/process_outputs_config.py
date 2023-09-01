import argparse
import json
import os
import ast
import matplotlib.pyplot as plt


def process_doc_event(document, tanl_info, gtt_info, types_mapping, create_tanl_template_input=False, create_tanl_template_output=False):
    error_analysis = {
        "docid": gtt_info["docid"],
        "doctext": gtt_info["doctext"],
        "gold_templates": gtt_info["templates"],
        "pred_templates": []
    }

    trig_to_template = {}
    template_pairs = {}
    if create_tanl_template_input or create_tanl_template_output:
        template_pairs = {
            "tokens": tanl_info["tokens"],
            "inputs": {"entities": [], "triggers": [], "relations": []},
            "outputs": {"entities": [], "triggers": [], "relations": []},
            "id": gtt_info["docid"]
        }
    for trig_tup in document["triggers"]:
        incident_type = types_mapping[trig_tup[0]]
        new_trig_tup = (incident_type, trig_tup[-2], trig_tup[-1])
        trig_to_template[new_trig_tup] = {
            "incident_type": incident_type,
            "PerpInd": [],
            "PerpOrg": [],
            "Target": [],
            "Victim": [],
            "Weapon": []
        }

        if create_tanl_template_input:
            template_pairs["inputs"]["triggers"].append({
                "type": trig_tup[0],
                "start": trig_tup[-2],
                "end": trig_tup[-1]
            })
        if create_tanl_template_output:
            template_pairs["outputs"]["triggers"].append({
                "type": trig_tup[0],
                "start": trig_tup[-2],
                "end": trig_tup[-1]
            })

    for arg_tup in document["args"]:
        arg_type = types_mapping[arg_tup[0]]
        arg_span = " ".join(tanl_info["tokens"][arg_tup[1][0]: arg_tup[1][1]])
        trigger_tup = (types_mapping[arg_tup[-1][0]],
                       arg_tup[-1][-2], arg_tup[-1][-1])
        trig_to_template[trigger_tup][arg_type].append([arg_span])

        if create_tanl_template_input:
            entity_tup_tanl = {
                "type": "template entity",
                "start": arg_tup[1][0],
                "end": arg_tup[1][1]
            }
            trig_tup_tanl = {
                "type": arg_tup[-1][0],
                "start": arg_tup[-1][-2],
                "end": arg_tup[-1][-1]
            }

            if not entity_tup_tanl in template_pairs["inputs"]["entities"]:
                template_pairs["inputs"]["entities"].append(entity_tup_tanl)
            template_pairs["inputs"]["relations"].append({
                "type": arg_tup[0],
                "head": template_pairs["inputs"]["entities"].index(entity_tup_tanl),
                "tail": template_pairs["inputs"]["triggers"].index(trig_tup_tanl)
            })

    if create_tanl_template_output:
        for relation in tanl_info["relations"]:
            ref_ent = tanl_info["entities"][relation["head"]]
            if not ref_ent in template_pairs["outputs"]["entities"]:
                template_pairs["outputs"]["entities"].append(ref_ent)
            
            template_pairs["outputs"]["relations"].append({
                "type": relation["type"],
                "head": template_pairs["outputs"]["entities"].index(ref_ent),
                "tail": template_pairs["outputs"]["triggers"].index(tanl_info["triggers"][relation["tail"]])
            })

    for template in trig_to_template.values():
        error_analysis["pred_templates"].append(template)

    return error_analysis, template_pairs

def process_doc_ner(document, tanl_info, gtt_info):
    error_analysis = {
        "docid": gtt_info["docid"],
        "tokens": tanl_info["tokens"],
        "pred_triggers": list(document["triggers"]),
        "pred_args": list(document["args"]),
        "gold_triggers": [],
        "gold_args": []
    }

    for template in gtt_info["templates"]:
        for role, entities in template.items():
            if role != "incident_type":
                for coref_set in entities:
                    error_analysis["gold_triggers"] += coref_set

def handle_buffer_event(buffer, tanl_infos, gtt_infos, types_mapping, create_tanl_template_input=False, create_tanl_template_output=False):
    results = []
    tanl_template_pairs = []
    document = {
        "id": None,
        "triggers": set(),
        "args": set()
    }
    reference_count = 0
    for i, line in enumerate(buffer):
        if line.startswith("id"):
            id = line[3:17].strip()  # maybe have to check indexing

            if document["id"] != None and document["id"] != id:
                tanl_info, gtt_info = tanl_infos[reference_count], gtt_infos[reference_count]
                assert tanl_info["id"].strip() == document["id"]
                assert gtt_info["docid"].strip() == document["id"]

                result, tanl_template_pair = process_doc_event(
                    document, tanl_info, gtt_info, types_mapping, create_tanl_template_input, create_tanl_template_output)
                results.append(result)
                tanl_template_pairs.append(tanl_template_pair)

                document = {
                    "id": id,
                    "triggers": set(),
                    "args": set()
                }
                reference_count += 1
            else:
                document["id"] = id

        elif line.startswith("triggers"):
            # maybe have to check indexing
            document["triggers"] = document["triggers"].union(
                set(ast.literal_eval(line[9:-1])))

        else:
            # maybe have to check indexing
            document["args"] = document["args"].union(
                {tuple(item) for item in ast.literal_eval(line[10:])})

    return results, tanl_template_pairs

def handle_buffer_ner(buffer, tanl_infos, gtt_infos, types_mapping, create_tanl_template_input=False, create_tanl_template_output=False):
    results = []
    tanl_template_pairs = []
    document = {
        "id": None,
        "triggers": set(),
        "args": set()
    }
    reference_count = 0
    for i, line in enumerate(buffer):
        if line.startswith("id"):
            id = line[3:17].strip()  # maybe have to check indexing

            if document["id"] != None and document["id"] != id:
                tanl_info, gtt_info = tanl_infos[reference_count], gtt_infos[reference_count]
                assert tanl_info["id"].strip() == document["id"]
                assert gtt_info["docid"].strip() == document["id"]

                result, tanl_template_pair = process_doc_ner(
                    document, tanl_info, gtt_info, types_mapping, create_tanl_template_input, create_tanl_template_output)
                results.append(result)
                tanl_template_pairs.append(tanl_template_pair)

                document = {
                    "id": id,
                    "triggers": set(),
                    "args": set()
                }
                reference_count += 1
            else:
                document["id"] = id

        else:
            extracted_entities = set(ast.literal_eval(line[10:-1]))
            document["triggers"].union({(tup[1], tup[2]) for tup in extracted_entities if tup[0] == 'trigger' and tup[1] < tup[2]})
            document["args"].union({(tup[1], tup[2]) for tup in extracted_entities if tup[0] == 'event argument' and tup[1] < tup[2]})

    return results, tanl_template_pairs

def process_text(mode, outs, tanl_infos, gtt_infos, output_files, types_mapping, error_analysis, second_phase_args, log_input=False, log_output=False):
    results = []
    EVAL_PART, TEST_PART = "EVAL PART\n", "TEST PART\n"
    is_writing_eval = True
    buffer = []

    if isinstance(tanl_infos, list):
        with open(tanl_infos[0], "r") as f:
            reference_tanl = json.loads(f.read())
        with open(tanl_infos[1], "r") as f:
            reference_tanl_2 = json.loads(f.read())
    else:
        with open(tanl_infos, "r") as f:
            reference_tanl = json.loads(f.read())
    if isinstance(gtt_infos, list):
        with open(gtt_infos[0], "r") as f:
            reference_gtt = json.loads(f.read())
        with open(gtt_infos[1], "r") as f:
            reference_gtt_2 = json.loads(f.read())
    else:
        with open(gtt_infos, "r") as f:
            reference_gtt = json.loads(f.read())

    if isinstance(output_files, list):
        out_file, out_file_2 = output_files
    else:
        out_file = output_files

    with open(outs, "r") as f:
        in_lines = f.readlines()

    for line in in_lines:
        if is_writing_eval:
            if line == TEST_PART:
                if mode != "ner":
                    new_results, _ = handle_buffer_event(
                        buffer, reference_tanl, reference_gtt, types_mapping)

                    if isinstance(tanl_infos, list):
                        with open("temp.json", "w") as f:
                            f.write(
                                json.dumps({id: document for id, document in enumerate(new_results)}))
                        os.system(
                            'python {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file))
                else:
                    pass

                buffer = []
                is_writing_eval = False

            elif line != EVAL_PART:
                buffer.append(line)
        else:
            if line == EVAL_PART:
                if mode != "ner":
                    new_results, _ = handle_buffer_event(
                        buffer, reference_tanl_2, reference_gtt_2, types_mapping)

                    if isinstance(tanl_infos, list):
                        with open("temp.json", "w") as f:
                            f.write(
                                json.dumps({id: document for id, document in enumerate(new_results)}))
                        os.system(
                            'python {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file_2))

                buffer = []
                is_writing_eval = True
            elif line != TEST_PART:
                buffer.append(line)

    if isinstance(tanl_infos, list):
        if mode != "ner":
            results, _ = handle_buffer_event(
                buffer, reference_tanl_2, reference_gtt_2, types_mapping)
    else:
        if mode != "ner":
            results, tanl_template_pairs = handle_buffer_event(
                buffer, reference_tanl, reference_gtt, types_mapping, log_input, log_output)

    with open("temp.json", "w") as f:
        f.write(
            json.dumps({id: document for id, document in enumerate(results)}))

    if isinstance(tanl_infos, list):
        os.system(
            'python {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file_2))
    else:
        os.system(
            'python {} -i "temp.json" -o "{}" --verbose -s all -m "MUC_Errors" -at'.format(
                error_analysis, out_file)
        )
        if len(second_phase_args):
            if second_phase_args["training_data"]:
                with open(second_phase_args["training_data"], "w") as f:
                    f.write(json.dumps(tanl_template_pairs))

            if second_phase_args["second_phase_logging_train_output_file"]:
                with open(second_phase_args["second_phase_logging_train_output_file"], "w") as f:
                    tanl_template_pairs_unique_id = []
                    for template_pair in tanl_template_pairs:
                        if all(template_pair["id"] != existing_pair["id"] for existing_pair in tanl_template_pairs_unique_id):
                            tanl_template_pairs_unique_id.append(template_pair)
                        if len(tanl_template_pairs_unique_id) == second_phase_args["second_phase_logging_train_num"]:
                            break

                    f.write(json.dumps(tanl_template_pairs_unique_id))

            if second_phase_args["second_phase_logging_test_output_file"]:
                with open("temp_test.json", "r") as f:
                    test_samples = json.loads(
                        f.read())[:second_phase_args["second_phase_logging_test_num"]]
                    with open(second_phase_args["second_phase_logging_test_output_file"], "w") as f2:
                        f2.write(json.dumps(test_samples))

    if isinstance(log_input, str):
        with open(log_input, "w") as f:
            f.write(json.dumps([
                {
                    "entities": document["inputs"]["entities"],
                    "triggers": document["inputs"]["triggers"],
                    "relations": document["inputs"]["relations"],
                    "tokens": document["tokens"],
                    "id": document["id"]
                }
                for document in tanl_template_pairs]))
    os.remove("temp.json")
    if os.path.exists("_.out"):
        os.remove("_.out")


def main(config, mode):
    with open(config["types_mapping"], "r") as f:
        types_mapping = json.loads(f.read())

    if config["template_errors_train_out"]:  # calculate training muc losses
        process_text(
            mode,
            config["template_errors_train_out"],
            config["template_errors_train_tanl_ref"],
            config["template_errors_train_gtt_ref"],
            config["template_errors_train_output_files"],
            types_mapping,
            config["error_analysis_quick"],
            {},
        )
    if config["template_errors_test_out"]:  # calculate test muc losses
        process_text(
            mode,
            config["template_errors_test_out"],
            config["template_errors_test_tanl_ref"],
            config["template_errors_test_gtt_ref"],
            config["template_errors_test_output_file"],
            types_mapping,
            config["error_analysis_full"],
            {},
            "temp_test.json"
        )
    if config["second_phase_train_out"]:  # calculate muc losses and generate second phase data
        process_text(
            mode,
            config["second_phase_train_out"],
            config["second_phase_train_tanl_ref"],
            config["second_phase_train_gtt_ref"],
            config["second_phase_train_output_file"],
            types_mapping,
            config["error_analysis_full"],
            config["second_phase_args"],
            True,
            True
        )

    if os.path.exists("temp_test.json"):
        os.remove("temp_test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--config", type=str, required=False,
                        default="process_outputs.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.loads(f.read())[args.mode]

    main(config, args.mode)
