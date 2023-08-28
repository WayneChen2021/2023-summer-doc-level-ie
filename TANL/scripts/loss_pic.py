import json
import os
import matplotlib.pyplot as plt
import argparse

def main(dir):
    img_dir = os.path.join(dir, "images")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    for file in os.listdir(dir):
        if "json" in file:
            with open(os.path.join(dir, file), "r") as f:
                data = json.loads(f.read())
                ax = plt.gca()
                ax.scatter(
                    [i["step"] for i in data if "loss" in i],
                    [i["loss"] for i in data if "loss" in i],
                    s=0.01
                )
                ax.set_yscale('log')
                plt.savefig(os.path.join(img_dir, "{}.png".format(file[:-5])))
                plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.log_dir)