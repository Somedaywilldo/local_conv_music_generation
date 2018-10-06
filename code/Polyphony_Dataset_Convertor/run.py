import os

if __name__ == '__main__':
    rlaunch = ''
    for id in [3]:
        os.system(rlaunch + 'python3 polyphony_train_conv.py %d' % (id))