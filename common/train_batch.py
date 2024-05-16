import train_utils
from train_utils import default_arguments

import train_lora
from train_lora import train
import os


if __name__ == "__main__":
    people_folder = "/home/ubuntu/AutoLoRADiscovery/people"
    people = [os.path.join(people_folder,f) for f in os.listdir(people_folder)]
    os.makedirs("lora_outputs", exist_ok=True)

    for person in people:
        args = default_arguments
        args["instance_data_dir"] = person
        args["output_dir"] = "lora_outputs/lora-"+person.split("/")[-1]
        train(args)