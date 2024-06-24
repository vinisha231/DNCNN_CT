import argparse
from glob import glob

import tensorflow as tf
import pdb

from model import denoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--ndct_eval_data', dest='ndct_eval_set', default='.', help='ndct dataset for eval in training')
parser.add_argument('--ldct_eval_data', dest='ldct_eval_set', default='.', help='ldct dataset for eval in training')
parser.add_argument('--ndct_test_data', dest='ndct_test_set', default='.', help='ndct dataset for testing')
parser.add_argument('--ldct_test_data', dest='ldct_test_set', default='.', help='ldct dataset for testing')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--ndct_training_data', dest='ndct_training_data', default='.', help='ndct dataset for training')
parser.add_argument('--ldct_training_data', dest='ldct_training_data', default='.',help='ldct dataset for training')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
    with load_data(filepath=args.ndct_training_data) as ndct_data, load_data(filepath=args.ldct_training_data) as ldct_data:
        ldct_eval_files = sorted(glob(args.ldct_eval_set))
        #print(ldct_eval_files)
        ldct_eval_data = load_floats(ldct_eval_files)  # list of array of different size, 4-D, pixel value range is 0-255

        ndct_eval_files = sorted(glob(args.ndct_eval_set))
        #print(ndct_eval_files)
        ndct_eval_data = load_floats(ndct_eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
        denoiser.train(ndct_data, ldct_data, ndct_eval_data, ldct_eval_data, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr,
                       sample_dir=args.sample_dir)


def denoiser_test(denoiser):
    ldct_files= sorted(glob(args.ldct_test_set))
    ldct_files= load_floats(ldct_files)
    ndct_files= sorted(glob(args.ndct_test_set))
    ndct_files= load_floats(ndct_files)
    denoiser.test(ldct_files, ndct_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[30:] = lr[0] / 10.0
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, batch_size = args.batch_size)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
