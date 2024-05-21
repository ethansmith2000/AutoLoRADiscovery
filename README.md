# AutoLoRADiscovery

This repo houses a couple of projects centered around attempting to reduce the redundancy of model training, specifically in narrow domain areas, making use of few-parameter methods like LoRA training. An explanation can be found [here](https://sweet-hall-e72.notion.site/Automated-LoRA-Discovery-and-Teaching-Neural-Networks-to-make-Neural-Networks-22aa3b5ad66e4bc985ff2c93896538d2?pvs=4)

## Requirements
torch, diffusers, transformers, accelerate should do the trick
you may need bitsandbytes and insightface depending on your needs


## How To Use
in /common you will find train_lora.py as well as train_batch.py which can be used for training LoRAs, it's worth experimenting with how few parameters you can hit while still maintaining fidelity.

All scripts have arguments within the scripts themselves or in the common/train_utils.py which you can edit.

Alternatively, you can try the datset I've provided [here](https://huggingface.co/datasets/ethansmith2000/lora_bundle_celeb/tree/main) containing 136 LoRAs trained on celebrities. 

Once you have obtained all your LoRAs, you should use the provided notebook to aggregate all the LoRAs into a single list and regularize the weights using the singular value decomposition. Save this so you can then point to it with the other training scripts.

From there, a method that does consistently work is in PCLora, where you can train new LoRAs by learning coefficients weighing all of the provided LoRAs.

To get more experimental, you can try the discover_X methods which aim to train hypernetworks to produce LoRAs on the fly.
