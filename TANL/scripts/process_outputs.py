import argparse
import json
import ast
import os


def process_doc(document, tanl_info, gtt_info, types_mapping, create_tanl_template=False):
    error_analysis = {
        "docid": gtt_info["docid"],
        "doctext": gtt_info["doctext"],
        "gold_templates": gtt_info["templates"],
        "pred_templates": []
    }

    trig_to_template = {}
    template_pairs = {}
    if create_tanl_template:
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

        if create_tanl_template:
            template_pairs["inputs"]["triggers"].append({
                "type": trig_tup[0],
                "start": trig_tup[-2],
                "end": trig_tup[-1]
            })
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

        if create_tanl_template:
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

    if create_tanl_template:
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


def handle_buffer(buffer, tanl_infos, gtt_infos, types_mapping, create_tanl_template=False):
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
                # print(document["id"])
                # print(tanl_info["id"])
                assert tanl_info["id"].strip() == document["id"]
                assert gtt_info["docid"].strip() == document["id"]

                result, tanl_template_pair = process_doc(
                    document, tanl_info, gtt_info, types_mapping, create_tanl_template)
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


def main(in_file, out_file, out_file_2, reference_tanl, reference_gtt, reference_tanl_2, reference_gtt_2, types_mapping, error_analysis, multi_phase_out):
    with open(in_file, "r") as f:
        in_lines = f.readlines()

    with open(reference_tanl, "r") as f:
        tanl_infos = json.loads(f.read())

    with open(reference_gtt, "r") as f:
        gtt_infos = json.loads(f.read())

    if reference_tanl_2:
        with open(reference_tanl_2, "r") as f:
            tanl_infos_2 = json.loads(f.read())

    if reference_gtt_2:
        with open(reference_gtt_2, "r") as f:
            gtt_infos_2 = json.loads(f.read())

    with open(types_mapping, "r") as f:
        types_mapping = json.loads(f.read())

    results = []
    EVAL_PART, TEST_PART = "EVAL PART\n", "TEST PART\n"
    is_writing_eval = True
    buffer = []
    for i, line in enumerate(in_lines):
        if is_writing_eval:
            if line == TEST_PART:
                new_results, _ = handle_buffer(
                    buffer, tanl_infos, gtt_infos, types_mapping)

                if reference_tanl_2 and error_analysis != None:
                    with open("temp.json", "w") as f:
                        f.write(
                            json.dumps({id: document for id, document in enumerate(new_results)}))
                    os.system(
                        'python {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file))
                else:
                    results += new_results

                buffer = []
                is_writing_eval = False

            elif line != EVAL_PART:
                buffer.append(line)
        else:
            if line == EVAL_PART:
                new_results, _ = handle_buffer(
                    buffer, tanl_infos_2, gtt_infos_2, types_mapping)

                if reference_tanl_2 and error_analysis != None:
                    with open("temp.json", "w") as f:
                        f.write(
                            json.dumps({id: document for id, document in enumerate(new_results)}))
                    os.system(
                        'python {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file_2))
                else:
                    results += new_results

                buffer = []
                is_writing_eval = True
            elif line != TEST_PART:
                buffer.append(line)

    if reference_tanl_2:
        results, _ = handle_buffer(
            buffer, tanl_infos_2, gtt_infos_2, types_mapping)
    else:
        results, tanl_template_pairs = handle_buffer(
            buffer, tanl_infos, gtt_infos, types_mapping, multi_phase_out != None)

    with open("temp.json", "w") as f:
        f.write(
            json.dumps({id: document for id, document in enumerate(results)}))

    if reference_tanl_2:
        if error_analysis:
            os.system(
                'python {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file_2))
    else:
        if multi_phase_out:
            with open(multi_phase_out, "w") as f:
                f.write(json.dumps(tanl_template_pairs))

        if error_analysis:
            os.system(
                'python {} -i "temp.json" -o "{}" --verbose -s all -m "MUC_Errors" -at'.format(
                    error_analysis, out_file)
            )

    os.remove("temp.json")
    if os.path.exists("_.out"):
        os.remove("_.out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=False)
    parser.add_argument("--out_file_2", type=str, required=False)
    parser.add_argument("--reference_tanl", type=str, required=True)
    parser.add_argument("--reference_gtt", type=str, required=True)
    parser.add_argument("--reference_tanl_2", type=str,
                        required=False, default=None)
    parser.add_argument("--reference_gtt_2", type=str,
                        required=False, default=None)
    parser.add_argument("--types_mapping", type=str, required=True)
    parser.add_argument("--error_analysis", type=str, required=False)
    parser.add_argument("--multiphase_out", type=str, required=False)
    args = parser.parse_args()

    main(args.in_file, args.out_file, args.out_file_2, args.reference_tanl,
         args.reference_gtt, args.reference_tanl_2, args.reference_gtt_2, args.types_mapping, args.error_analysis, args.multiphase_out)
