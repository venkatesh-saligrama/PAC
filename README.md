# Pretraining and Consistency

This repository contains the implementation for the BMVC'21 paper 
"Surprisingly Simple Semi-Supervised Domain Adaptation with Pretraining and Consistency".

## Install

`pip install -r requirements.txt`

## Data preparation

Our implementation follows a similar data format to [MME](https://github.com/VisionLearningGroup/SSDA_MME).
To download and set up DomainNet data for the _real_ and _sketch_ domains, run

`sh download_data.sh`

For setting up data for _clipart_ and _painting_, uncomment corresponding lines in `download_data.sh`.
Images go into the following directories:

`./data/multi/real/<category_name>`,

`./data/multi/sketch/<category_name>`

And file lists in the txt format are available in the following directories:

`./data/txt/multi/labeled_source_images_real.txt`,

`./data/txt/multi/unlabeled_target_images_sketch_3.txt`,

`./data/txt/multi/validation_target_images_sketch_3.txt`.

Office, Office-Home and VisDA-17 datasets, can be set up in a similar manner, 
with filelists provided in their respective directories in `data/`.

## Training

For pretraining with rotation prediction

`python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=5001 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred`

For semi-supervised domain adaptation (SSDA) training

`python main.py --steps=50001 --dataset=multi --source=real --target=sketch --backbone=expts/rot_pred/checkpoint.pth.tar`


## Other experiments
The following are commands to run the experiments for the corresponding results in the paper.

- **Virtual Adversarial Training (VAT)** : Adversarial perturbation using VAT instead of image augmentation can be used as follows:

`python main.py --vat_tw=0.01 --steps=10001 --net=alexnet --aug_level=0 --dataset=office_home --source=Real --target=Clipart`

- **Pretraining with Momentum Contrast (MoCo)**

`python addnl_scripts/pretrain/moco.py --steps=5001 --dataset=office_home --source=Real --target=Clipart --save_dir=expts/moco_pretraining`

- **Evaluation scripts** : To compute the ![equation](https://bit.ly/3rr9OoM) -distance, use (while providing the correct `dataset`, `source`, `target` and `net`)
  
`python addnl_scripts/eval/compute_proxy_distance.py --backbone_path=path/to/backbone`

For computing nearest neighbors classifier accuracy, use 

`python addnl_scripts/eval/nn_classifier.py --backbone_path=path/to/backbone`

## Citation

If you find this repository useful for your work, please consider citing:
```
@article{mishra2021surprisingly,
  title={Surprisingly Simple Semi-Supervised Domain Adaptation with Pretraining and Consistency},
  author={Mishra, Samarth and Saenko, Kate and Saligrama, Venkatesh},
  journal={arXiv preprint arXiv:2101.12727},
  year={2021}
}
```
