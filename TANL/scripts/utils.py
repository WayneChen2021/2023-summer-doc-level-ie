import ast
import json
import os
import argparse
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt

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
            "docid": "DEV-MUC3-0300",
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


def add_annotations(examples, tanl_info, gtt_info):
    return [
        {
            "model_out": example,
            "tanl": tanl_info[i],
            "gtt": gtt_info[i]
        }
        for i, example in enumerate(examples)
    ]


def handle_buffer_ner(buffers):
    results = []
    document = {
        "id": None,
        "triggers": set(),
        "args": set()
    }
    for buffer in buffers:
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

            else:
                extracted_entities = set(ast.literal_eval(line[10:-1]))
                document["triggers"].union(
                    {(tup[1], tup[2]) for tup in extracted_entities if tup[0] == 'trigger' and tup[1] < tup[2]})
                document["args"].union(
                    {(tup[1], tup[2]) for tup in extracted_entities if tup[0] == 'event argument' and tup[1] < tup[2]})

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

            else:
                # maybe have to check indexing
                document["args"] = document["args"].union(
                    {tuple(item) for item in ast.literal_eval(line[10:])})

        splits.append(document)
        results.append(splits)

    return results


def split_to_multi_section(file):
    buffers = []
    buffer = []
    is_writing_eval = True
    EVAL_PART, TEST_PART = "EVAL PART\n", "TEST PART\n"

    with open(file, "r") as f:
        for line in f.readlines():
            if is_writing_eval:
                if line == TEST_PART:
                    buffers.append(buffer)
                    buffer = []
                    is_writing_eval = False
                elif line != EVAL_PART and len(line) > 1:
                    buffer.append(line)

            else:
                if line == EVAL_PART:
                    buffers.append(buffer)
                    buffer = []
                    is_writing_eval = True
                elif line != TEST_PART and len(line) > 1:
                    buffer.append(line)

    buffers.append(buffer)

    return buffers

def load_data_train(model_out, tanl_ref, gtt_ref):
    eval_tanl_ref, test_tanl_ref = tanl_ref
    with open(eval_tanl_ref, "r") as f:
        tanl_ref = json.loads(f.read())
    with open(test_tanl_ref, "r") as f:
        tanl_ref_2 = json.loads(f.read())
    
    eval_gtt_ref, test_gtt_ref = gtt_ref
    with open(eval_gtt_ref, "r") as f:
        gtt_ref = json.loads(f.read())
    with open(test_gtt_ref, "r") as f:
        gtt_ref_2 = json.loads(f.read())

    buffers = split_to_multi_section(model_out)

    return buffers, tanl_ref, tanl_ref_2, gtt_ref, gtt_ref_2

def load_data_test(model_out, tanl_ref, gtt_ref):
    with open(tanl_ref, "r") as f:
        tanl_ref = json.loads(f.read())
    
    with open(gtt_ref, "r") as f:
        gtt_ref = json.loads(f.read())

    buffers = split_to_multi_section(model_out)

    return buffers, tanl_ref, gtt_ref

def get_error_analysis_input(model_out, tanl_ref, gtt_ref, handle_buffer, to_error_analysis_format, types_mapping):
    if types_mapping:
        with open(types_mapping, "r") as f:
            types_mapping = json.loads(f.read())

    all_inputs = []
    if isinstance(tanl_ref, list):
        buffers, tanl_ref, tanl_ref_2, gtt_ref, gtt_ref_2 = load_data_train(model_out, tanl_ref, gtt_ref)
        is_eval = True
        for split in handle_buffer(buffers):
            if is_eval:
                triplets = add_annotations(split, tanl_ref, gtt_ref)
                is_eval = False
            else:
                triplets = add_annotations(split, tanl_ref_2, gtt_ref_2)
                is_eval = True
            
            error_analysis_input = to_error_analysis_format(triplets, types_mapping)
            all_inputs.append(error_analysis_input)
        
        return all_inputs
    else:
        buffers, tanl_ref, gtt_ref = load_data_test(model_out, tanl_ref, gtt_ref)
        triplets = add_annotations(handle_buffer(buffers)[0], tanl_ref, gtt_ref)
        error_analysis_input = to_error_analysis_format(triplets, types_mapping)

        return [error_analysis_input]


def error_analysis_event(model_out, tanl_ref, gtt_ref, error_analysis, types_mapping, out_file):
    error_analysis_inputs = get_error_analysis_input(model_out, tanl_ref, gtt_ref, handle_buffer_event, to_error_analysis_format_event, types_mapping)
    
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
    error_analysis_inputs = get_error_analysis_input(model_out, tanl_ref, gtt_ref, handle_buffer_ner, to_error_analysis_format_ner, None)
    
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


def plot_training_errors(error_analysis_summary_train, error_analysis_summary_test, loss_log, eval_interval, loss_interval, out_file):
    summary_x = []
    f1_entries_train, recall_entries_train, precision_entries_train = [], [], []
    f1_entries_test, recall_entries_test, precision_entries_test = [], [], []
    with open(error_analysis_summary_train, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            if len(line) > 1:
                info = json.loads(line)["total"]
                summary_x.append(i * eval_interval)
                f1_entries_train.append(info["f1"])
                recall_entries_train.append(info["recall"])
                precision_entries_train.append(info["precision"])
    
    with open(error_analysis_summary_test, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            if len(line) > 1:
                info = json.loads(line)["total"]
                f1_entries_test.append(info["f1"])
                recall_entries_test.append(info["recall"])
                precision_entries_test.append(info["precision"])
    
    loss_x = []
    loss_entries = []
    with open(loss_log, "r") as f:
        for i, log in enumerate(json.loads(f.read()), 1):
            if "loss" in log:
                loss_x.append(i * loss_interval)
                loss_entries.append(log["loss"])
    
    plt.title("training statistics")
    plt.xlabel("steps")
    plt.ylabel("amount")
    plt.yscale("log")
    plt.plot(summary_x, f1_entries_train, label = "train f1")
    plt.plot(summary_x, recall_entries_train, label = "train recall")
    plt.plot(summary_x, precision_entries_train, label = "train precision")
    plt.plot(summary_x, f1_entries_test, label = "test f1")
    plt.plot(summary_x, recall_entries_test, label = "test recall")
    plt.plot(summary_x, precision_entries_test, label = "test precision")
    plt.plot(loss_x, loss_entries, label = "train loss")
    plt.legend()
    plt.savefig(out_file)
    plt.clf()

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
            plot_training_errors(
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
            plot_training_errors(
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
    
    multi_tasks = ["multi_phase", "verbose_tags", "multiple_triggers", "focused_cross_entropy_typed", "focused_cross_entropy_notype"]
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
        }
    }
    for job_category in multi_tasks:
        jobs = config[job_category]
        for job_name in ["test_time_logs", "train_time_logs", "training_errors"]:
            if jobs and job_name in jobs and jobs[job_name]["run"]:
                for job_id, single_config in parse_multiple_jobs(jobs[job_name], multi_job_fields[job_category][job_name]).items():
                    run_task(jobs["mode"], job_name, single_config, jobs["types_mapping"], job_id, jobs)