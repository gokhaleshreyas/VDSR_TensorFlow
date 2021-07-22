import vdsr
import tensorflow as tf
import argparse

train_dir = '/train/'
test_dir = '/test/Set5'
validation_dir = '/test/Set5'
model_dir = '/model/'
result_dir = '/result/'

args = argparse.ArgumentParser()
args.add_argument('--do_train', type=bool, default=False)
args.add_argument('--do_test', type=bool, default=False)
args.add_argument('--train_dir', type=str, default=train_dir)
args.add_argument('--valid_dir', type=str, default=validation_dir)
args.add_argument('--test_dir', type=str, default=test_dir)
args.add_argument('--model_dir', type=str, default=model_dir)
args.add_argument('--result_dir', type=str, default=result_dir)
args.add_argument('--scale', type=int, default=3)
args.add_argument('--learning_rate', type=float, default=1e-4)
args.add_argument('--momentum', type=float, default=0.9)
args.add_argument('--epochs', type=int, default=100)
args.add_argument('--n_channels', type=int, default=1)
args.add_argument('--batch_size', type=int, default=128)
args.add_argument('--colour_format', type=str, default='ych')
args.add_argument('--depth', type=int, default=20)
args.add_argument('--prepare_data', type=str, default='matlab')


def main(args):
    with tf.Session() as sess:
        net = vdsr.VDSR(args, sess)
        if args.do_train:
            net.train()
        elif args.do_test:
            net.test()
        else:
            print("Invalid value for --do_train or --do_test")


if __name__ == '__main__':
    main(args=args.parse_args())
