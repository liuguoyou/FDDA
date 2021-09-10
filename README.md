# Fine-grained Data Distribution Alignment for Post-Training Quantization ([Paper Link](https://arxiv.org/abs/2109.04186)) 

## Requirements

- Python >= 3.7.10
- Pytorch = 1.2.0 (For resnet18/mobilenet_w1/mobilenet_w2), Pytorch = 1.7.1 (For regnetx_600m)
- Torchvision >= 0.4.0

## Reproduce the Experiment Results 

1. The pre-trained model will be downloaded automatically. If the download process fails, please use the URL in the console to download manually.

2. Randomly select one image per class to generate corresponding BNS centers, selected images will be formulating the calibration dataset, run:
    
    `cd FDDA`

    `mkdir save_ImageNet`

    `CUDA_VISIBLE_DEVICES=0 python BNScenters.py --dataPath PathToImageNetDataset --model_name resnet18/mobilenet_w1/mobilenet_w2/regnetx_600m`  
   
   Noted that a model will generate corresponding BNS centers that can't be used by other model.

4. Use FDDA to train a quantized model. Modify the `qw, qa` in imagenet_config.hocon to set desired bit-width. Modify the `dataPath` in imagenet_config.hocon to the path of ImageNet Dataset. For all layers are quantized to same bit-width, run:

    `CUDA_VISIBLE_DEVICES=0 python main_cosine_CBNS.py --model_name resnet18/mobilenet_w1/mobilenet_w2/regnetx_600m --conf_path imagenet_config.hocon --id=0`

   For F8L8, run:
   
   `CUDA_VISIBLE_DEVICES=0 python main_cosine_CBNS_8F8L.py --model_name resnet18/mobilenet_w1/mobilenet_w2/regnetx_600m --conf_path imagenet_config.hocon --id=0`

## Evaluate Our Models

We also provide training logs and trained models for test. They can be downloaded from [here](https://drive.google.com/drive/folders/1LNhxoYKG2fz3D3-7A7WiMpdjAh8f-HZH?usp=sharing):

To test our models, download it and run:

   `CUDA_VISIBLE_DEVICES=0 python test.py --conf_path imagenet_config_.hocon --model_name resnet18/mobilenet_w1/mobilenet_w2/regnetx_600m --model_path PathToModel`
