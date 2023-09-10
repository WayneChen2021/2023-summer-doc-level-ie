import json
from matplotlib import pyplot as plt

def add_annotations(examples, tanl_info, gtt_info):
    if gtt_info:
        return [
            {
                "model_out": example,
                "tanl": tanl_info[i],
                "gtt": gtt_info[i]
            }
            for i, example in enumerate(examples)
        ]

    return [
        {
            "model_out": example,
            "tanl": tanl_info[i]
        }
        for i, example in enumerate(examples)
    ]

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
    if tanl_ref:
        with open(tanl_ref, "r") as f:
            tanl_ref = json.loads(f.read())
    
    if gtt_ref:
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

def create_second_phase_train(model_out, tanl_ref, out_file, handle_buffer, create_annotation):
    buffers, tanl_ref, _ = load_data_test(model_out, tanl_ref, None)
    tuples = add_annotations(handle_buffer(buffers)[0], tanl_ref, None)
    
    with open(out_file, "w") as f:
        f.write(json.dumps([create_annotation(tup["model_out"], tup["tanl"]) for tup in tuples]))