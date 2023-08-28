import os
import json
import argparse
from process_raw_outputs import run_permutation

eval_part, test_part = "EVAL PART\n", "TEST PART\n"

def eval(model_outs, tanl_file, gtt_file, output_file, edge_cases, start_from, error_analysis):
    global eval_part
    global test_part

    if start_from != 0:
        open(output_file, "w").close()
    
    with open(model_outs, "r") as f:
        lines = list(f.readlines())
        cut = []
        begin_add = True
        for i, line in enumerate(lines[start_from:lines.index("END OF TRAINING\n")], start_from):
            if line == eval_part:
                begin_add = False
            elif line == test_part:
                if len(cut):
                    open("temp.txt", "w").close()
                    with open("temp.txt", "a") as f:
                        for single_line in cut:
                            f.write(single_line)
                    
                    print("Reached {} line {}".format(model_outs, i))
                    
                    run_permutation(-1, "temp.txt", tanl_file, gtt_file, "temp.out", "", edge_cases)
                    os.system('python {} -i "temp.out" -o "_.out" -s all -m "MUC_Errors" --muc_errors "{}"'.format(error_analysis, output_file))
                    
                    cut = []
                else:
                    begin_add = True
            else:
                if begin_add:
                    cut.append(line)
    
    os.remove("_.out")
    os.remove("temp.out")
    os.remove("temp.txt")

def train(model_outs, tanl_file, gtt_file, output_file, edge_cases, start_from, error_analysis):
    global eval_part
    global test_part

    if start_from != 0:
        open(output_file, "w").close()
    
    with open(model_outs, "r") as f:
        lines = list(f.readlines())
        cut = []
        begin_add = True
        for i, line in enumerate(lines[start_from:lines.index("END OF TRAINING\n")], start_from):
            if line == eval_part:
                if len(cut):
                    open("temp.txt", "w").close()
                    with open("temp.txt", "a") as f:
                        for single_line in cut:
                            f.write(single_line)
                    
                    print("Reached {} line {}".format(model_outs, i))
                    
                    run_permutation(-1, "temp.txt", tanl_file, gtt_file, "temp.out", "", edge_cases)
                    os.system('python3 {} -i "temp.out" -o "_.out" -s all -m "MUC_Errors" --muc_errors "{}"'.format(error_analysis, output_file))
                    
                    cut = []
                    begin_add = True
            elif line == test_part:
                begin_add = False
            else:
                if begin_add:
                    cut.append(line)
    
    os.remove("_.out")
    os.remove("temp.out")
    os.remove("temp.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--is_eval", type=bool, required=True)
    parser.add_argument("--model_outs", type=str, required=True)
    parser.add_argument("--tanl_file", type=str, required=True)
    parser.add_argument("--gtt_file", type=str, required=True)
    parser.add_argument("--edge_cases", type=str, required=False)
    parser.add_argument("--start_from", type=int, required=False, default=0)
    parser.add_argument("--error_analysis", type=str, required=False, default="Error_Analysis_quick.py")
    args = parser.parse_args()
    
    edge_cases = {}
    if args.edge_cases:
        with open(args.edge_cases, "r") as f:
            edge_cases = json.loads(f.read())
    
    if args.is_eval:
        eval(args.model_outs, args.tanl_file, args.gtt_file, args.output_file, edge_cases, args.start_from, args.error_analysis)
    else:
        train(args.model_outs, args.tanl_file, args.gtt_file, args.output_file, edge_cases, args.start_from, args.error_analysis)