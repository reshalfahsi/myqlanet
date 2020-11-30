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

import myqlanet

# define the network
model = myqlanet.MyQLaNet()

# predict from the given path to weight and its root path
result = model.predict("/path/to/weight","/root/path/")

# print the result in the form of bounding box data: y lower, x, lower, y upper, x upper 
print(result)

```

Example output:

```
(15, 100, 135, 220)
```
