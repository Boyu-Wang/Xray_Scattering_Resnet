# Resnet model for x-ray scattering images

This is tensorflow implementation for using resnet for xray scattering image classification.

## Prerequirement
The model is fully test on tensorflow 1.4. 

For other requirement, please check requirements.txt

## Data
Data can be downloaded [here](https://drive.google.com/drive/folders/1y5i5YlWdA9KC6mM0p9LL1IfmAR1uwFs4?usp=sharing).  The raw.zip contains generated synthetic images. The tfrecords.zip contains TFRecords files required to train the network. The convert_to_tfrecords.py can process the raw data and convert them to TFRecords. You can also download the processed files. 

## Running

To train the model, please run:

```bash
python train.py
```

One pretrained model can be download [here](https://drive.google.com/drive/folders/1y5i5YlWdA9KC6mM0p9LL1IfmAR1uwFs4?usp=sharing). Put it under save/resnet folder.

To load pretrained model and test on your own images, please check 

```bash
ipython notebook test.ipynb
```


## License
GPL3

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{Wang-etal-WACV17, 
Author = {Boyu Wang and Kevin Yager and Dantong Yu and Minh Hoai}, 
Booktitle = {Proceedings of Winter Conference on Applications of Computer Vision}, 
Title = {X-ray Scattering Image Classification Using Deep Learning}, 
Year = {2017}} 
```

# Notes
the residual network implementation is based on [here](https://github.com/ry/tensorflow-resnet)

