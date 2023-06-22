🔧 Requirements
=
- Python 3
- PyTorch


- Check the required python packages in `requirements.txt`.
```
pip install -r requirements.txt
```

Data and Data Prepareing Tools
=
The Datasets we used:
  * [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
  * [DISFA](http://mohammadmahoor.com/disfa-contact-form/)

We provide tools for prepareing data in ```tool/```.
After Downloading raw data files, you can use these tools to process them, aligning with our protocals.
More details have been described in [tool/README.md](tool/README.md).


**Training with ImageNet pre-trained models**

Make sure that you download the ImageNet pre-trained models to `checkpoints/` (or you alter the checkpoint path setting in `models/resnet.py` or `models/swin_transformer.py`)

The download links of pre-trained models are in `checkpoints/checkpoints.txt`

Thanks to the offical Pytorch and [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

Training and Testing
=

- to train our approach (ResNet-50) on BP4D Dataset with Gated-GCN and only vertex feeding to final classifier, run:
```
python train_link_prediction.py --dataset BP4D --arc resnet50 --exp-name resnet50_link_prediction  --resume results/resnet50_link_prediction/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 --gnn_type GCN --feed_type vertex  --root_dir .
```

- to train our approach (Swin-B) on BP4D Dataset with GAT and both edge and vertex feeding to final classifier, run:
```
python train_link_prediction.py --dataset BP4D --arc resnet50 --exp-name swin_base_link_prediction --resume results/swin_base_link_prediction/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 --gnn_type GAT --feed_type vertex+edge  --root_dir .
```

- to train our approach (Swin-B) on DISFA Dataset with Gated-GCN and edge feeding to final classifier, run:
```
python train_link_prediction.py --dataset DISFA --arc resnet50 --exp-name swin_base_link_prediction  --resume results/swin_base_link_prediction/bs_64_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1 --lam 0.05 --gnn_type GAT --feed_type edge    --root_dir .
```

- to test the performance on DISFA Dataset (for Swin-B, Gated-GCN and edge feed), run:
```
python test_link_prediction.py --dataset DISFA --arc swin_transformer_base --exp-name test_fold2  --resume results/swin_transformer_base_link_prediction/bs_64_seed_0_lr_0.000001/xxxx_fold2.pth --fold 2 --gnn_type GCN --feed_type edge  --root_dir .
```


### Pretrained models

BP4D

TBA

DISFA

TBA



📝 Main Results
=
**BP4D**

TBA

**DISFA**

TBA
