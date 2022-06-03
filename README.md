# 3DCV Final project Group 15

Memeber: D0892202 楊証琨, R09946018 葉柏宏
Advisor: Prof. 陳祝嵩



## Environment

- Main

```bash
# Torch
$ pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
# MinkowskiEngine 0.4.1
$ conda install numpy openblas
$ git clone https://github.com/StanfordVL/MinkowskiEngine.git
$ cd MinkowskiEngine
$ git checkout f1a419cc5792562a06df9e1da686b7ce8f3bb5ad
$ python setup.py install
# Others
$ pip install imageio==2.8.0 opencv-python==4.2.0.32 pillow==7.0.0 pyyaml==5.3 scipy==1.4.1 sharedarray==3.2.0 tensorboardx==2.0 tqdm==4.42.1
```
- Others

    Please refer to [env.yml](./env.yml) for details.

## Prepare data

- Download the dataset from official website.

- 2D: The scripts is from 3DMV repo, it is based on python2, other code in this repo is based on python3
	```python prepare_2d_data.py --scannet_path data/scannetv2 --output_path data/scannetv2_images --export_label_images```
	
- 3D: dataset/preprocess_3d_scannet.py

## Config
- BPNet_5cm: config/scannet/weak_bpnet_5cm.yaml 

## Training
- Download pretrained 2D ResNets on ImageNet from PyTorch website, and put them into the `initmodel` folder.
```python
model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}
```
- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**

## Testing

- Testing using our [weakly-supervised BPNet](https://drive.google.com/file/d/1POi7pPM79E1GTr0fx5TGaluLjVJPMDc1/view?usp=sharing) (voxel_size: 5cm):
```sh tool/test.sh weak_bpnet ./EXP/scannet/weak_bpnet_5cm.yaml NUMBER_OF_THREADS)```



