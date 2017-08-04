# Universal Training Control Tools 

A set of utilities and wrappers for uniform work with various frameworks such as Tensorflow, TFLearn, Keras, MXNet, Lasagne, and CNTK. This allows us to apply universal methods for hyper parameters tuning, overfitting detection, etc. Also, it simplifies porting/converting models from one framework to another.

## Instructions for using TFLearn-branch on Linux (GPU):
1. Install TensorFlow (https://www.tensorflow.org/install/install_linux):
```
sudo pip install tensorflow-gpu
```
2. Install TFLearn (http://tflearn.org/installation):
```
sudo pip install git+https://github.com/tflearn/tflearn.git
```
3. Install extra python packages:
- Bayesian optimization:
```
sudo pip install bayesian-optimization
```
- OpenCV:
```
sudo pip install opencv_python
```
## Instructions for using TFLearn-branch on Win10 x64 (CPU):
1. Install Python3:
- Install Anaconda3 x64 for Windows (https://www.continuum.io/downloads).
- Add the path to Anaconda for ability of running the python from a console:
```
PATH=%PATH%; %USERPROFILE%\Anaconda3
```
2. Install TensorFlow (https://www.tensorflow.org/install/install_windows):
```
pip install --upgrade tensorflow
```
3. Install TFLearn (http://tflearn.org/installation):
```
pip install git+https://github.com/tflearn/tflearn.git
```
4. Install extra python packages:
- Bayesian optimization:
```
pip install bayesian-optimization
```
- OpenCV (from http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv):
```
pip install opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl
```

## Instructions for using MXNet-branch on Linux (GPU):
1. Install MXNet (http://mxnet.io/get_started/install.html):
```
sudo apt-get update
sudo apt-get install -y wget python
wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
sudo pip install mxnet-cu80
```
2. Install extra python packages:
- Graphviz:
```
sudo apt-get install graphviz
sudo pip install graphviz
```
- Bayesian optimization:
```
sudo pip install bayesian-optimization
```
- OpenCV:
```
sudo pip install opencv_python
```

## Instructions for using MXNet-branch on Win10 x64 (CPU):
1. Install Python3:
- Install Anaconda3 x64 for Windows (https://www.continuum.io/downloads).
- Add the path to Anaconda for ability of running the python from a console:
```
PATH=%PATH%; %USERPROFILE%\Anaconda3
```
2. Install MXNet:
- Download MXNet binary packages for Windows x64 (https://github.com/yajiedesign/mxnet/releases).
- Unpack packages into `%LANGS_LIBS%/mxnet` directory.
- Setup environmental variables (from script `setupenv.cmd`, NB: This bugged script slightly breaks existing paths!):
```
set MXNET_HOME=%LANGS_LIBS%\mxnet
set PATH=%PATH%; %MXNET_HOME%\3rdparty\openblas\bin
set PATH=%PATH%; %MXNET_HOME%\3rdparty\gnuwin
set PATH=%PATH%; %MXNET_HOME%\3rdparty\vc
set PATH=%PATH%; %MXNET_HOME%\3rdparty\opencv
set PATH=%PATH%; %MXNET_HOME%\3rdparty\cudart
set PATH=%PATH%; %MXNET_HOME%\3rdparty\cudnn\bin
set PATH=%PATH%; %MXNET_HOME%\lib
```
- Install the MXNet python package by running `python setup.py install`.
- Copy content of folder `%MXNET_HOME%\python\mxnet` to `%USERPROFILE%\Anaconda3\Lib\site-packages\mxnet-0.10.1-py3.6.egg\mxnet`.
3. Install extra python packages:
- Bayesian optimization:
```
pip install bayesian-optimization
```
- OpenCV (from http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv):
```
pip install opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl
```
4. Install Graphviz (http://www.graphviz.org/Download_windows.php) and add path to the program in to the environmental variables:
```
set PATH=%PATH%; C:\Program Files (x86)\Graphviz2.38\bin
```
