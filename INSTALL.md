# VideoMAE Installation

The codebase is mainly built with following libraries:

- Python 3.6 or higher

- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). <br>
  We can successfully reproduce the main results under two settings below:<br>
  Tesla **A100** (40G): CUDA 11.1 + PyTorch 1.8.0 + torchvision 0.9.0<br>
  Tesla **V100** (32G): CUDA 10.1 + PyTorch 1.6.0 + torchvision 0.7.0

- [timm==0.4.8/0.4.12](https://github.com/rwightman/pytorch-image-models)
  `conda install -c conda-forge timm=0.4.12`

- [deepspeed==0.5.8](https://github.com/microsoft/DeepSpeed)

  `DS_BUILD_OPS=1 pip install deepspeed`

  **Comment**: I couldn't install it with this flag. Without it, the installation was completed successfully with the newest stable Pytorch. In case of an error related to ds_kernels, install this first: 

  `pip install deepspeed-kernels` [DeepSpeed-Kernels repo](https://github.com/microsoft/DeepSpeed-Kernels)

- [TensorboardX](https://github.com/lanpa/tensorboardX)

- [decord](https://github.com/dmlc/decord)

- [einops](https://github.com/arogozhnikov/einops)

- `pip install tensorboardX decord einops opencv-python scipy pandas`
- `pip install scikit-learn matplotlib seaborn torchmetrics`

### Note:
- We recommend you to use **`PyTorch >= 1.8.0`**.
- We observed accidental interrupt in the last epoch when conducted the pre-training experiments on V100 GPUs (PyTorch 1.6.0). This interrupt is caused by the scheduler of learning rate. We naively set  `--epochs 801` to walk away from issue :)

