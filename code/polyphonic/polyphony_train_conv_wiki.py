from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical


import time
import numpy as np
import random
import sys
import os
from math import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import LambdaCallback

from my_to_midi import *

import pickle as pkl

from polyphony_dataset_convertor import *
import time

start_time = time.time()
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 256, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('epochs', 150, 'Total epochs')
tf.app.flags.DEFINE_integer('maxlen', 64, 'Max length of a sentence')
tf.app.flags.DEFINE_integer('generate_length', 3200,
                            'Number of steps of generated music')
tf.app.flags.DEFINE_integer('units', 512, 'LSTM Layer Units Number')
tf.app.flags.DEFINE_integer('dense_size', 0, 'Dense Layer Size')
tf.app.flags.DEFINE_integer('step', 32, 'Step length when building dataset')
tf.app.flags.DEFINE_integer('embedding_length', 1, 'Embedding length')
tf.app.flags.DEFINE_string('dataset_name', 'Wikifonia',
                           'Dataset name will be the prefix of exp_name')
tf.app.flags.DEFINE_string('dataset_dir', './datasets/',
                           'Dataset Directory, which should contain name_train.txt and name_eval.txt')


batch_size = FLAGS.batch_size
epochs = FLAGS.epochs
units = FLAGS.units
dense_size = FLAGS.dense_size


maxlen = FLAGS.maxlen
generate_length = FLAGS.generate_length
step = FLAGS.step
embedding_length = FLAGS.embedding_length
dataset_name = FLAGS.dataset_name
dataset_dir = FLAGS.dataset_dir


date_and_time = time.strftime('%Y-%m-%d_%H%M%S')


exp_name = 'CNN_64TS_32Step_Wiki'
vector_dim = 259


train_dataset_path = '/home/ouyangzhihao/sss/Mag/Mag_Data/Poly/Wikifonia/Wikifonia_new_train.pkl'
eval_dataset_path = '/home/ouyangzhihao/sss/Mag/Mag_Data/Poly/Wikifonia/Wikifonia_new_eval.pkl'

with open(train_dataset_path, "rb") as train_file:
    train_data = pkl.load(train_file)
    '''
    temp = []
    for i in train_data:
        temp = temp + i[1:len(i)-1]
    '''
    train_data = np.array(train_data)
    train_file.close()

print('Train dataset shape:', train_data.shape)

with open(eval_dataset_path, "rb") as eval_file:
    eval_data = pkl.load(eval_file)
    '''
    temp = []
    for i in eval_data:
        temp = temp + i[1:len(i)-1]
    '''
    eval_data = np.array(eval_data)
    eval_file.close()

print('Eval dataset shape:', eval_data.shape)


log_root = './log'
log_dir = os.path.join(log_root, "logdir", exp_name)
TB_log_dir = os.path.join(log_root, 'TB_logdir', exp_name)
console_log_dir = os.path.join(log_root, log_dir, "console")
model_log_dir = os.path.join(log_root, 'Model_logdir', exp_name)
data_log_dir = os.path.join(log_root, log_dir, "data")
midi_log_dir = os.path.join(log_root, log_dir, "midi")


def make_log_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


dirs = [log_dir, TB_log_dir, console_log_dir,
        model_log_dir, data_log_dir, midi_log_dir]
make_log_dirs(dirs)

max_acc_log_path = os.path.join(log_root, "logdir", "max_acc_log.txt")


def get_embedded_data(data, maxlen, embedding_length):

    inputs = np.array(data[:len(data) - 1])
    labels = np.array(data[1:len(data)])

    print(np.shape(inputs))
    print(np.shape(inputs[0]))
    print((inputs[0]))

    inputs = to_categorical(inputs, 259)
    labels = to_categorical(labels, 259)

    inputs_emb = inputs
    label_emb = labels

    inputs_maxlen = []
    label_maxlen = []
    for i in range(0, len(inputs_emb) - maxlen, step):
        inputs_maxlen.append((inputs_emb[i: i + maxlen]))
        label_maxlen.append(label_emb[i+maxlen])

    return np.asarray(inputs_maxlen, dtype=np.float16), np.asarray(label_maxlen, dtype=np.float16)


train_data = train_data[:int(len(train_data)/1)]
eval_data = eval_data[:int(len(eval_data)/1)]

print('Vectorization...')
x_train, y_train = get_embedded_data(train_data, maxlen, embedding_length)
x_eval, y_eval = get_embedded_data(eval_data, maxlen, embedding_length)


def print_fn(str):
    print(str)
    console_log_file = os.path.join(console_log_dir, 'console_output.txt')
    with open(console_log_file, 'a+') as f:
        print(str, file=f)


def lr_schedule(epoch):

    lr = 1e-1
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print_fn('Learning rate: %f' % lr)

    lr = 1e-3
    return lr


print_fn('Build model...')


train_output_shape = vector_dim
train_input_shape = (maxlen, vector_dim)
from ConvModel import *
from ConvResModel import *
from ConvOtherStructureModel import *
from keras import backend as K


model = get_conv1d_model(input_shape=train_input_shape,
                         output_shape=train_output_shape)


def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)

    return perplexity


optimizer = Adam(lr=lr_schedule(0))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy', perplexity])
model.summary(print_fn=print_fn)


def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_music(epoch, data, diversity, start_index, is_train=False):
    print_fn('----- diversity: %.1f' % diversity)

    generated = [0]
    events = data[start_index: start_index + maxlen]
    generated += events
    print('----- Generating with seed: ', events)
    print(generated)

    generated = list(generated)

    for i in range(generate_length):
        x_pred = np.zeros((1, maxlen, 259 * embedding_length))
        for t, event in enumerate(events):

            x_pred[0, t, event % 259] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_event = int(next_index)

        generated.append(next_event)
        events = events[1:] + [next_event]

    if is_train:
        log_name = "epoch%d_train_diversity%02d" % (
            epoch + 1, int(diversity * 10))
    else:
        if start_index == 0:
            log_name = "epoch%d_first_diversity%02d" % (
                epoch + 1, int(diversity * 10))
        else:
            log_name = "epoch%d_random_diversity%02d" % (
                epoch + 1, int(diversity * 10))

    generated += [1]

    data_log_path = os.path.join(data_log_dir, log_name + ".pkl")
    with open(data_log_path, "wb") as data_log_file:
        data_log_file.write(pkl.dumps(generated))
        data_log_file.close()

    print_fn("Write %s.pkl to %s" % (log_name, data_log_dir))

    list_to_midi(generated, 120, midi_log_dir, log_name)

    print_fn("Write %s.midi to %s" % (log_name, midi_log_dir))

    model_name = "epoch%d.h5" % (epoch+1)
    model_path = os.path.join(model_log_dir, model_name)
    model.save(model_path)
    print_fn("Save model %s.h5 to %s" % (model_name, model_log_dir))


def on_epoch_end(epoch, logs):

    if (epoch+1) % (epochs // 5) != 0:
        return
    elif(epoch <= epochs * 3 / 5):
        return

    print_fn("")
    print_fn('----- Generating Music after Epoch: %d' % epoch)

    start_index = random.randint(0, len(train_data) - maxlen - 1)

    for diversity in [0.5, 0.8, 1.0, 1.2]:

        generate_music(epoch=epoch, data=train_data,
                       diversity=diversity, start_index=start_index, is_train=True)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


tb_callbacks = LRTensorBoard(log_dir=TB_log_dir)

print_fn("*"*20+exp_name+"*"*20)
print_fn('x_train shape:'+str(np.shape(x_train)))
print_fn('y_train shape:'+str(np.shape(y_train)))

history_callback = model.fit(x_train, y_train,
                             validation_data=(x_eval, y_eval),
                             verbose=1,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=[tb_callbacks, lr_scheduler, print_callback])


acc_history = history_callback.history["acc"]
max_acc = np.max(acc_history)
print_fn('Experiment %s max accuracy:%f' % (exp_name, max_acc))
max_acc_log_line = "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%f" % (exp_name,
                                                       epochs, units, dense_size, maxlen, step, embedding_length, max_acc)

print(max_acc_log_line, file=open(max_acc_log_path, 'a'))
print('Total time Cost:', time.time() - start_time)
