import argparse
import json

def main(model_out, gtt_file):
    with open(model_out, "r") as f:
        model_out_lines = f.readlines()
    
    with open(gtt_file, "r") as f:
        gtt_docs = json.loads(f.read())
    
    

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