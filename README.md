# Invisible Poison: A Blackbox Clean Label Attack to Deep Neural Networks

This repository provides the code for the paper 
<br>
[Invisible Poison: A Blackbox Clean Label Attack to Deep Neural Networks](http://)
<br>

### Citation

## Dependencies
* [Python 3.6+ ](https://www.python.org)
* [PyTorch 1.0+](http://pytorch.org)
* [TorchVision](https://www.python.org)
* [scipy](https://www.scipy.org)
* [matplotlib](https://matplotlib.org/#)
* [TQDM](https://github.com/tqdm/tqdm)
* [sklearn](http://scikit-learn.github.io/stable)

## Getting Started
---

### Installation 

1. Clone this repository
```bash
https://github.com/rigley007/Invi_Poison.git
```

2. Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision, tqdm, Numpy)

3. Download the dataset from [here](https://www.dropbox.com/s/r90p4hx8wiczog7/data.zip). Then you need to unzip them and change the path in config.py accordingly.

### Usages
+ To train the auto-encoder to convert original image to noised image, you can run the script as follows:
```bash
python3 main.py
```
The model will save checkpoint every 20 epochs on the fly.

+ To train the auto-encoder to reconstruct image from noised images, you can run the script with:
```bash
python3 autodecoder_training.py
```

+ You also can experiment with varying setting by editing configuration files in `configs.py`. 
