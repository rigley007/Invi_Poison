# Invisible Poison: A Blackbox Clean Label Backdoor Attack to Deep Neural Networks

This repository provides the code for the paper 
<br>
[Invisible Poison: A Blackbox Clean Label Backdoor Attack to Deep Neural Networks](http://)
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

3. Download the dataset from [here](https://drive.google.com/file/d/1NrEOREa3FtQ1TTLAUtQQEuSU_PPHMXs4/view?usp=sharing). Then you need to unzip them and change the path in config.py accordingly.

### Usages
+ To train the auto-encoder to convert original image to noised image, you can run the script as follows:
```bash
python3 main.py
```
The model will save a checkpoint every 20 epochs on the fly.

+ To train the auto-encoder to reconstruct images from noised images, you can run the script with:
```bash
python3 autodecoder_training.py
```

+ To train and test the target model with poisoned data injected (demostration of the attack), run the script with:
```bash
python3 training_with_poisoned_dataset.py
```

+ You also can experiment with different settings by editing configuration files in `configs.py`. 
