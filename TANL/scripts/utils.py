import ast
import json
import os
import argparse
import BaseProcessing
from tqdm import tqdm
from copy import deepcopy

def to_error_analysis_format_event(triplets, types_mapping):
    outputs = {}
    for triplet in triplets:
        error_analysis = {
            "docid": triplet["gtt"]["docid"],
            "doctext": triplet["gtt"]["doctext"],
            "gold_templates": triplet["gtt"]["templates"],
            "pred_templates": []
        }

        trig_to_template = {}
        for trig_tup in triplet["model_out"]["triggers"]:
            try:
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
            except KeyError:
                print("type '{}' not identified".format(trig_tup[0]))
        
        assert error_analysis["docid"].strip() == triplet["tanl"]["id"].strip()
        assert triplet["tanl"]["id"].strip() == triplet["model_out"]["id"].strip()

        for arg_tup in triplet["model_out"]["args"]:
            try:
                arg_type = types_mapping[arg_tup[0]]
                arg_span = " ".join(triplet["tanl"]["tokens"][arg_tup[1][0]: arg_tup[1][1]])
                trigger_tup = (types_mapping[arg_tup[-1][0]],
                            arg_tup[-1][-2], arg_tup[-1][-1])
                trig_to_template[trigger_tup][arg_type].append([arg_span])
            except KeyError:
                print("types '{}' or '{}' not identified".format(arg_tup[0], arg_tup[-1][0]))
        
        for template in trig_to_template.values():
            error_analysis["pred_templates"].append(template)
        
        outputs[error_analysis["docid"]] = error_analysis
    
    return outputs


def to_error_analysis_format_ner(triplets, types_mapping):
    outputs = []
    for triplet in triplets:
        error_analysis = {
            "docid": triplet["model_out"]["id"],
            "tokens": triplet["tanl"]["tokens"],
            "pred_triggers": list(triplet["model_out"]["triggers"]),
            "pred_args": list(triplet["model_out"]["args"]),
            "gold_triggers": list(set((info["start"], info["end"]) for info in triplet["tanl"]["triggers"])),
            "gold_args": [],
        }

        for template in triplet["gtt"]["templates"]:
            for role, entities in template.items():
                if role != "incident_type":
                    for coref_set in entities:
                        error_analysis["gold_args"].append(coref_set)

        outputs.append(error_analysis)
    
    return outputs


def handle_buffer_ner(buffers):
    results = []
    for buffer in buffers:
        document = {
            "id": None,
            "triggers": set(),
            "args": set()
        }
        splits = []
        for line in buffer:
            if line.startswith("id"):
                id = line[3:17].strip()  # maybe have to check indexing

                if document["id"] != None and document["id"] != id:
                    splits.append(deepcopy(document))
                    document = {
                        "id": id,
                        "triggers": set(),
                        "args": set()
                    }
                else:
                    document["id"] = id

            elif line.startswith("arguments"):
                extracted_entities = set(ast.literal_eval(line[10:-1]))
                document["triggers"] = document["triggers"].union(
                    {(tup[1], tup[2]) for tup in extracted_entities if tup[0] == 'trigger' and tup[1] < tup[2]})
                document["args"] = document["args"].union(
                    {(tup[1], tup[2]) for tup in extracted_entities if tup[0] == 'event argument' and tup[1] < tup[2]})

        splits.append(document)
        results.append(splits)

    return results


def handle_buffer_event(buffers):
    results = []
    for buffer in buffers:
        document = {
            "id": None,
            "triggers": set(),
            "args": set()
        }
        splits = []
        for line in buffer:
            if line.startswith("id"):
                id = line[3:17].strip()  # maybe have to check indexing

                if document["id"] != None and document["id"] != id:
                    splits.append(deepcopy(document))
                    document = {
                        "id": id,
                        "triggers": set(),
                        "args": set()
                    }
                else:
                    document["id"] = id
            elif line.startswith("triggers"):
                # maybe have to check indexing
                document["triggers"] = document["triggers"].union(
                    set(ast.literal_eval(line[9:])))

            elif line.startswith("arguments"):
                # maybe have to check indexing
                document["args"] = document["args"].union(
                    {tuple(item) for item in ast.literal_eval(line[10:])})

        splits.append(document)
        results.append(splits)

    return results


def error_analysis_event(model_out, tanl_ref, gtt_ref, error_analysis, types_mapping, out_file):
    error_analysis_inputs = BaseProcessing.get_error_analysis_input(model_out, tanl_ref, gtt_ref, handle_buffer_event, to_error_analysis_format_event, types_mapping)
    
    if isinstance(out_file, list):
        eval_out, test_out = out_file
        is_eval = True
        for inputs in tqdm(error_analysis_inputs, desc="Processing train time template errors..."):
            with open("temp.json", "w") as f:
                f.write(json.dumps(inputs))
            if is_eval:
                os.system('python3 {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, eval_out))
                is_eval = False
            else:
                os.system('python3 {} -i "temp.json" -o "_.out" --muc_errors "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, test_out))
                is_eval = True
    else:
        with open("temp.json", "w") as f:
            f.write(json.dumps(error_analysis_inputs[0]))
        
        os.system('python3 {} -i "temp.json" -o "{}" --verbose -s all -m "MUC_Errors" -at'.format(error_analysis, out_file))
    
    os.remove("temp.json")
    if os.path.exists("_.out"):
        os.remove("_.out")


def error_analysis_ner(model_out, tanl_ref, gtt_ref, error_analysis, out_file):
    error_analysis_inputs = BaseProcessing.get_error_analysis_input(model_out, tanl_ref, gtt_ref, handle_buffer_ner, to_error_analysis_format_ner, None)
    
    if isinstance(out_file, list):
        eval_out, test_out = out_file
        is_eval = True
        for inputs in tqdm(error_analysis_inputs, desc="Processing train time template errors..."):
            with open("temp.json", "w") as f:
                f.write(json.dumps(inputs))
            if is_eval:
                os.system('python3 {} --i "temp.json" --o "{}" --relax'.format(error_analysis, eval_out))
                is_eval = False
            else:
                os.system('python3 {} --i "temp.json" --o "{}" --relax'.format(error_analysis, test_out))
                is_eval = True
    else:
        with open("temp.json", "w") as f:
            f.write(json.dumps(error_analysis_inputs[0]))

        os.system('python3 {} --i "temp.json" --o "{}" --relax'.format(error_analysis, out_file))
    
    os.remove("temp.json")


def create_annotation_ner(tup):
    pred, tanl = tup["model_out"], tup["tanl"]
    return {
        "id": tanl["id"],
        "ner": {"triggers": [
            {"start": tup[0], "end": tup[1]}
            for tup in pred["triggers"]
        ], "arguments": [
            {"start": tup[0], "end": tup[1]}
            for tup in pred["args"]
        ]},
        "event": {"entities": tanl["entities"], "triggers": tanl["triggers"], "relations": tanl["relations"]},
        "tokens": tanl["tokens"]
    }

def create_annotation_event(tup):
    pred, tanl = tup["model_out"], tup["tanl"]
    example = {
        "entities": [],
        "triggers": [
            {
                "type": trig_tup[0],
                "start": trig_tup[1],
                "end": trig_tup[2]
            }
            for trig_tup in pred["triggers"]
        ],
        "relations": [],
        "tokens": tanl["tokens"],
        "id": tanl["id"]
    }
    for arg_tup in pred["args"]:
        entity = {
            "type": "template entity",
            "start": arg_tup[1][0],
            "end": arg_tup[1][1]
        }
        if not entity in example["entities"]:
            example["entities"].append(entity)
        
        trigger = {
            "type": arg_tup[-1][0],
            "start": arg_tup[-1][1],
            "end": arg_tup[-1][2]
        }
        example["relations"].append({
            "type": arg_tup[0],
            "head": example["entities"].index(entity),
            "tail": example["triggers"].index(trigger)
        })
    
    return example

def second_phase_training_event(model_out, tanl_ref, gtt_ref):
    pass

def second_phase_training_ner(model_out, tanl_ref, gtt_ref):
    pass

def run_task(mode, log_name, config, types_mapping=None, id=None, global_config=None):
    if mode == "event":
        if log_name in ["train_time_logs", "test_time_logs"]:
            error_analysis_event(
                config["raw_outs"],
                config["tanl_ref"],
                config["gtt_ref"],
                config["error_analysis_script"],
                types_mapping,
                config["output_file"]
            )
        else:
            error_analysis_summary = global_config["train_time_logs"]["output_file"][id]
            BaseProcessing.plot_training_errors(
                error_analysis_summary[0],
                error_analysis_summary[1],
                config["log_file"],
                config["small_evaluation_interval"],
                config["loss_collection_interval"],
                config["output_file"]
            )

    else:
        if log_name in ["train_time_logs", "test_time_logs"]:
            error_analysis_ner(
                config["raw_outs"],
                config["tanl_ref"],
                config["gtt_ref"],
                config["error_analysis_script"],
                config["output_file"]
            )
        else:
            error_analysis_summary = global_config["train_time_logs"]["output_file"][id]
            BaseProcessing.plot_training_errors(
                error_analysis_summary[0],
                error_analysis_summary[1],
                config["log_file"],
                config["small_evaluation_interval"],
                config["loss_collection_interval"],
                config["output_file"]
            )


def parse_multiple_jobs(job, multi_job_fields):
    configs = {}
    config = {k: v for k, v in job.items() if not k in multi_job_fields}

    for job_id in job[multi_job_fields[0]].keys():
        for field in multi_job_fields:
            config[field] = job[field][job_id]
        
        configs[job_id] = deepcopy(config)

    return configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.loads(f.read())
    
    multi_tasks = ["multi_phase_ner", "multi_phase_event", "verbose_tags", "multiple_triggers", "focused_cross_entropy_typed", "focused_cross_entropy_notype"]
    multi_job_fields = {
        "verbose_tags": {
            "test_time_logs": ["raw_outs", "output_file"],
            "train_time_logs": ["raw_outs", "output_file"],
            "training_errors": ["log_file", "output_file"]
        },
        "multiple_triggers": {
            "test_time_logs": ["raw_outs", "output_file"],
            "train_time_logs": ["raw_outs", "output_file"],
            "training_errors": ["log_file", "output_file"]
        },
        "focused_cross_entropy_typed": {
            "test_time_logs": ["raw_outs", "output_file"],
            "train_time_logs": ["raw_outs", "output_file"],
            "training_errors": ["log_file", "output_file"]
        },
        "focused_cross_entropy_notype": {
            "test_time_logs": ["raw_outs", "output_file"],
            "train_time_logs": ["raw_outs", "output_file"],
            "training_errors": ["log_file", "output_file"]
        },
        "multi_phase_ner": {
            "test_time_logs": ["raw_outs", "output_file", "tanl_ref", "gtt_ref"],
            "train_time_logs": ["raw_outs", "output_file"],
            "training_errors": ["log_file", "output_file"]
        },
        "multi_phase_event": {
            "test_time_logs": ["raw_outs", "output_file", "tanl_ref", "gtt_ref"],
            "train_time_logs": ["raw_outs", "output_file"],
            "training_errors": ["log_file", "output_file"]
        }
    }
    for job_category in multi_tasks:
        jobs = config[job_category]
        for job_name in ["test_time_logs", "train_time_logs", "training_errors"]:
            if jobs and job_name in jobs and jobs[job_name]["run"]:
                for job_id, single_config in parse_multiple_jobs(jobs[job_name], multi_job_fields[job_category][job_name]).items():
                    run_task(jobs["mode"], job_name, single_config, jobs["types_mapping"], job_id, jobs)