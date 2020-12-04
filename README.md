[![Build Status](https://travis-ci.com/reshalfahsi/myqlanet.svg?token=VeywGWKntUx4TytDJzYs&branch=master)](https://travis-ci.org/reshalfahsi/myqlanet)
# MyQLaNet

<div align="center">
  <img src="https://i.ibb.co/K0qkr9g/MyQLaNet.png" width = 200>
</div>


A Deep Learning Platform for Macula Detection.

It provides end to end system for macula detection with graphical user interface.

![alt text](resources/img/gui.jpg)


## Dependencies

* Ubuntu Linux (OS)
* PyQt5 (GUI)
* Python > 3.x (Programming Language)
* PyTorch (Machine Learning Framework)
* OpenCV and scikit-image (Computer Vision Framework)


## Installation

~~~
sudo apt install pyqt5-dev-tools
sudo pip3 install -r requirements.txt --no-cache-dir
python3 app.py
~~~


## Working with the Library

Instead of using GUI, you can code from the scratch:

```python

from myqlanet import *

# path to the important config file
dataset_path = '/path/to/dataset'
annotation_path = '/path/to/annotation'
weight_path = '/path/to/weight'

# create MyQLaNet model
model = MyQLaNet()

# create dataset
dataset = MaculaDataset(annotation_path, dataset_path)

# training time!
model.compile(dataset)
model.fit(weight_path)

```