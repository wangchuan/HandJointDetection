import pytoolkit.files as fp
from hand_model import HandModel
import os

def generate_dataset():
    def convert(filename, fd, lb):
        path, name, ext = fp.fileparts(filename)
        filename = os.path.join('./', fd, name + ext)
        return (filename, lb)
    path = '../BAA_Dataset/'
    fds = ['04', '05', '06', '07']
    pairs = []
    for lb, fd in enumerate(fds):
        subpath = os.path.join(path, fd)
        filelist_jpg = fp.dir(subpath, '.jpg')
        pairs += [convert(f, fd, lb) for f in filelist_jpg]

    from random import shuffle
    shuffle(pairs)
    N = len(pairs)
    ratio = 0.7
    train_data = pairs[:int(N*ratio)]
    valid_data = pairs[int(N*ratio):]

    with open(os.path.join(path, 'train_data.txt'), 'w') as f:
        for datum in train_data:
            f.write(datum[0] + ' ' + str(datum[1]) + '\n')
    with open(os.path.join(path, 'valid_data.txt'), 'w') as f:
        for datum in valid_data:
            f.write(datum[0] + ' ' + str(datum[1]) + '\n')

# ------------------------------------------------------------------ #
def train_script():
    hm = HandModel('aam')
    train_data_path = 'G:/DataSets/hand/trainset/'
    hm.train(train_data_path)

def valid_script():
    hm = HandModel('aam')
    valid_data_path = 'G:/DataSets/hand/validset/'
    hm.validate(valid_data_path)

def compute_average_shape_script():
    hm = HandModel('aam')
    train_data_path = 'G:/DataSets/hand/trainset/'
    hm.compute_average_shape(train_data_path)

"""
    hm = HandModel('aam')
    train_data_path = 'G:/DataSets/hand/trainset/'
    valid_data_path = 'G:/DataSets/hand/validset/'
    hm.train(train_data_path)
    hm.compute_average_shape(train_data_path)
    hm.validate(valid_data_path)
"""