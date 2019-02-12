# Multispectral conditional Generative Adversarial Nets
This repository is an implementation of ["Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets"](https://arxiv.org/abs/1710.04835).

![Results](images/results.png)

## Requirements
I recommend Anaconda to manage your Python libraries.  
Because it is easy to install some of the libraries necessary to prepare the data.  

* Python3 (tested with 3.5.4)
* Chainer (tested with 5.0.0)
* cupy (tested with 5.0.0)
* matplotlib (tested with 2.2.2)
* OpenCV (tested with 3.3.1)
* tqdm (tested with 4.15.0)
* PyYAML (tested with 3.12)
* mpi4py (tested with 3.0.0)

## Preparing the data
Please refer to [make_dataset/README.md](make_dataset/README.md).

## Training examples
You need set each parameters in a config file.  
```
CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path configs/config_nirrgb2rgbcloud.yml --results_dir results/pix2pix
```
If you want to resume the training from snapshot, use `--snapshot` option.

* pretrained model (WIP)

## Evaluation examples
```
CUDA_VISIBLE_DEVICES=0 python test.py --dir_nir <path to nir dir> --dir_rgb <path to rgb dir> --imlist_nir <path to nir list file> --imlist_rgb <path to rgb list file> --results_dir results/test_pix2pix --config_path results/pix2pix/config_nirrgb2rgbcloud.yml --gen_model results/pix2pix/Generator_<iterations>.npz
```

## License
Academic use only.
