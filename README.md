# local_conv_music_generation

Music Generation with Local Connected Convolutional Neural Network.

Developed by: Zhihao Ouyang, Yihang Yin, Kun Yan.

Demo of this project is released at [Demo Link](https://somedaywilldo.github.io/local_conv_music_generation/).

## Set Up the Environment

Required Python3 Packages:

```
keras
python-rtmidi
pretty-midi
progressbar
```

We've writen these in script, for your convenience, just do this in shell:

```shell
$ sh init.sh
```
## Train the Monophony Model
Run the following command to launch experiment.

```shell
$ cd monophony
$ python3 monophony.py <model_id>
```
Replace the **<model_id>** to change the model for comparison.

```
    0 for 'conv1_model_a'
    1 for 'conv1_model_b'
    2 for 'conv1_model_c'
    3 for 'conv1_model_naive'
    4 for 'conv1_model_naive_big'
    5 for 'resnet_model_naive'
    6 for 'resNet_model_local'
    7 for 'LSTM_model'
```

## Train the Polyphony Model
### First Step
Download the training data from this [link](https://drive.google.com/open?id=18205S7ut3MEq9A3aiKS2tpY06Y7Khq3E)

Then extract the data into `./polyphonic/datasets`
### Second Step
Run the following command to launch experiment.
```shell
$ cd polyphony
$ python3 polyphony.py <model_id>
```
Replace the **<model_id>** to change the model for comparison

```
    0 for 'conv1_model_a'
    1 for 'conv1_model_b'
    2 for 'conv1_model_c'
    3 for 'conv1_model_naive'
    4 for 'conv1_model_naive_big'
    5 for 'resnet_model_naive'
    6 for 'resNet_model_local'
    7 for 'LSTM_model'
```



## Contact

