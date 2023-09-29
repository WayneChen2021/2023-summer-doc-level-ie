import argparse
import json

def main(in_file, out_file):
    with open(in_file, "r") as f:
        info = json.loads(f.read())
    
    relation_types = {}
    for role_types in info.values():
        for role in role_types:
            for other_role in role_types:
                if other_role != role:
                    relation_name = "same event {} and {}".format(role, other_role)
                    relation_types[relation_name] = {"verbose": relation_name}

    with open(out_file, "w") as f:
        f.write(json.dumps(
            {
                "entities": {
                    "template entity": {"verbose": "template entity"}
                },
                "relations": relation_types
            }
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    main(args.in_file, args.out_file)