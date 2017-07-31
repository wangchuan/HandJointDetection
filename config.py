import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 24, "batch size for training")
tf.flags.DEFINE_integer("image_size", 224, "input image size")
tf.flags.DEFINE_integer("epoches", 100, "number of epoches")
tf.flags.DEFINE_integer("disp", 1, "how many iterations to display")
tf.flags.DEFINE_float("weight_decay", 0.00, "weight decay")
tf.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.flags.DEFINE_string("data_path", "../BAA_Dataset/", "data path storing npy files")
tf.flags.DEFINE_string("log_path", "./log/", "log path storing checkpoints")
tf.flags.DEFINE_string("mode", "train", "train or test")

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True