# local_conv_music_generation
Music Generation with Local Connected Convolutional Neural Network

We prepare a demo page to introduce our work:

[Demo Link](https://somedaywilldo.github.io/local_conv_music_generation/)

Also all the codes are released at:

[Code Link](https://github.com/Somedaywilldo/local_conv_music_generation/tree/master/code)


# Local-conv Network in Music Experiment
## dependency
```
cd code
sh init.sh
```
## monophonic music model train
run the following command to launch experiment

```
cd code/monophonic
python3 monophony.py {model id}
```
replace the model id to change the model for comparison

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

## polyphonic music model train
### First Step
Download the training data from this [link](https://drive.google.com/open?id=18205S7ut3MEq9A3aiKS2tpY06Y7Khq3E)

Then extract the data into `./polyphonic/datasets`
### Second Step
run the following command to launch experiment
```
cd code/polyphonic
python3 polyphony.py {model id}
```
replace the model id to change the model for comparison

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


