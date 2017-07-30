from hand_model import HandModel

def main():
    hm = HandModel('aam')
    train_data_path = 'G:/DataSets/hand/trainset/'
    valid_data_path = 'G:/DataSets/hand/validset/'
    hm.train(train_data_path)
    hm.compute_average_shape(train_data_path)
    hm.validate(valid_data_path)

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

if __name__ == "__main__":
    valid_script()