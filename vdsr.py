import tensorflow as tf
import numpy as np
import os
import h5py
import time
from PIL import Image


class VDSR:
    def __init__(self, args, sess):
        self.sess = sess
        self.do_train = args.do_train
        self.do_test = args.do_test
        self.train_dir = args.train_dir
        self.test_dir = args.test_dir
        self.valid_dir = args.valid_dir
        self.model_dir = args.model_dir
        self.result_dir = args.result_dir
        self.scale = args.scale
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.epochs = args.epochs
        self.n_channels = args.n_channels
        self.batch_size = args.batch_size
        self.colour_format = args.colour_format
        self.depth = args.depth
        self.prepare_data = args.prepare_data
        if self.colour_format == 'ych':
            self.model_name = 'vdsr_ych'
        elif self.colour_format == 'ycbcr':
            self.model_name = 'vdsr_ycbcr'
        elif self.colour_format == 'rgb':
            self.model_name = 'vdsr_rgb'
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_channels])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_channels])
        self.weights = {
            'w1': tf.get_variable(initializer=tf.initializers.he_normal(),
                                  shape=[3, 3, self.n_channels, 64],
                                  dtype=tf.float32,
                                  name='w1'),
            'w{:d}'.format(self.depth):
                tf.get_variable(initializer=tf.initializers.he_normal(),
                                shape=[3, 3, 64, self.n_channels],
                                dtype=tf.float32,
                                name='w{:d}'.format(self.depth)),
        }
        self.biases = {
            'b1': tf.get_variable(initializer=tf.initializers.constant(0),
                                  shape=[64],
                                  dtype=tf.float32,
                                  name='b1'),
            'b{:d}'.format(self.depth):
                tf.get_variable(initializer=tf.initializers.constant(0),
                                shape=[self.n_channels],
                                dtype=tf.float32,
                                name='b{:d}'.format(self.depth))
        }
        for i in range(2, self.depth):
            self.weights['w{:d}'.format(i)] = tf.get_variable(
                initializer=tf.initializers.he_normal(),
                shape=[3, 3, 64, 64],
                dtype=tf.float32,
                name='w{:d}'.format(i))
            self.biases['b{:d}'.format(i)] = tf.get_variable(
                initializer=tf.initializers.constant(0),
                shape=[64],
                dtype=tf.float32,
                name='b{:d}'.format(i))

        self.output = self.model()
        self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.y))
        self.result = tf.clip_by_value(self.output, clip_value_min=0., clip_value_max=1.)
        self.saver = tf.train.Saver()
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        vars = []
        for i in range(1, self.depth):
            vars.append(self.weights['w{:d}'.format(i)])
            vars.append(self.biases['b{:d}'.format(i)])
        grads_and_vars = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                    momentum=self.momentum).compute_gradients(loss=self.loss)
        # grads_and_vars = [(tf.clip_by_value(i, clip_value_min= -i/self.learning_rate,
        #                                     clip_value_max=i/self.learning_rate),
        #                    j) for i,j in grads_and_vars]
        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=self.momentum).apply_gradients(grads_and_vars=grads_and_vars)

    def model(self):
        conv = tf.nn.conv2d(input=self.X,
                            filters=self.weights['w1'],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        conv = tf.nn.bias_add(conv, self.biases['b1'])
        conv = tf.nn.relu(conv)

        for i in range(2, self.depth):
            conv = tf.nn.conv2d(input=conv,
                                filters=self.weights['w{:d}'.format(i)],
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            conv = tf.nn.bias_add(conv, self.biases['b{:d}'.format(i)])
            conv = tf.nn.relu(conv)

        conv = tf.nn.conv2d(input=conv,
                            filters=self.weights['w{:d}'.format(self.depth)],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        conv = tf.nn.bias_add(conv, self.biases['b{:d}'.format(self.depth)])

        output = tf.add(conv, self.X)
        return output

    def train(self):
        print("Training Will Start Shortly")
        if self.prepare_data == 'matlab':
            train_X, train_y = load_matlab_data(self.train_dir, self.colour_format)
            valid_X_bc, valid_y_bc = make_matlab_bc_data(self.valid_dir, self.scale, self.colour_format)
            valid_y_gt = make_matlab_gt_data(self.valid_dir, self.colour_format)
        elif self.prepare_data == 'octave':
            train_X, train_y = load_octave_data(self.train_dir, self.colour_format)
            valid_X_bc, valid_y_bc = make_octave_bc_data(self.valid_dir, self.scale, self.colour_format)
            valid_y_gt = make_octave_gt_data(self.valid_dir, self.colour_format)
        else:
            print("Invalid arguments for prepare_data")
        start_time = time.time()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.model()
        total_batches = len(train_X) // self.batch_size
        batch_size = self.batch_size
        for i in range(self.epochs):
            loss = 0
            if i % 5 == 0 and i > 0:
                self.learning_rate = self.learning_rate * 0.1
                print("Learning Rate: {}".format(self.learning_rate))
            a = start_time
            b = time.time()
            for j in range(total_batches):
                batch_X = train_X[j * batch_size:(j + 1) * batch_size]
                batch_y = train_y[j * batch_size:(j + 1) * batch_size]
                _, batch_error = self.sess.run([self.optimizer, self.loss],
                                               feed_dict={self.X: batch_X, self.y: batch_y})
                loss = loss + batch_error

            valid_psnr = []
            bicubic_psnr = []
            for k in range(len(valid_X_bc)):
                h1, w1, c1 = valid_X_bc[k].shape
                h2, w2, c2 = valid_y_bc[k].shape
                h3, w3, c3 = valid_y_gt[k].shape
                if self.n_channels == 1:
                    valid_X_bc_ych = valid_X_bc[k][:, :, 0]
                    valid_y_bc_ych = valid_y_bc[k][:, :, 0]
                    valid_y_gt_ych = valid_y_gt[k][:, :, 0]
                    valid_X_bc_ych = valid_X_bc_ych.reshape([1, h1, w1, 1])
                    valid_y_bc_ych = valid_y_bc_ych.reshape([1, h2, w2, 1])
                    valid_y_gt_ych = valid_y_gt_ych.reshape([1, h3, w3, 1])
                    results = self.sess.run(self.result, feed_dict={self.X: valid_X_bc_ych, self.y: valid_y_gt_ych})
                    valid_y_bc_ych = valid_y_bc_ych[0]
                    valid_y_gt_ych = valid_y_gt_ych[0]
                    results = results[0]
                    bicubic_psnr.append(psnr(valid_y_bc_ych, valid_y_gt_ych))
                    valid_psnr.append(psnr(results, valid_y_gt_ych))
                elif self.n_channels == 3:
                    valid_X_bc_ = valid_X_bc[k]
                    valid_y_bc_ = valid_y_bc[k]
                    valid_y_gt_ = valid_y_gt[k]
                    valid_X_bc_ = valid_X_bc_.reshape([1, h1, w1, c1])
                    valid_y_bc_ = valid_y_bc_.reshape([1, h2, w2, c2])
                    valid_y_gt_ = valid_y_gt_.reshape([1, h3, w3, c3])
                    results = self.sess.run(self.result, feed_dict={self.X: valid_X_bc_, self.y: valid_y_gt_})
                    valid_y_bc_ = valid_y_bc_[0]
                    valid_y_gt_ = valid_y_gt_[0]
                    results = results[0]
                    bicubic_psnr.append(psnr(valid_y_bc_, valid_y_gt_))
                    valid_psnr.append(psnr(results, valid_y_gt_))
                else:
                    print("Invalid Argument for n_channels")
            print(f"Epoch: {i + 1}, Bicubic PSNR: {np.mean(bicubic_psnr)}, VDSR PSNR: {np.mean(valid_psnr)}"
                  f"Time: {b - a}")
        self.save()
        print("Training Complete")
        end_time = time.time()
        print("Time Taken: {}".format(end_time - start_time))

    def test(self):
        print("Testing will commence")
        if self.prepare_data == 'matlab':
            test_X_bc, test_y_bc = make_matlab_bc_data(self.test_dir, self.scale, self.colour_format)
            test_y_gt = make_matlab_gt_data(self.test_dir, self.colour_format)
        elif self.prepare_data == 'octave':
            test_X_bc, test_y_bc = make_octave_bc_data(self.test_dir, self.scale, self.colour_format)
            test_y_gt = make_octave_gt_data(self.test_dir, self.colour_format)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.load()
        colour_format = self.colour_format
        bicubic_psnr = []
        test_psnr = []
        start_time = time.time()
        for i in range(len(test_X_bc)):
            h1, w1, c1 = test_X_bc[i].shape
            h2, w2, c2 = test_y_bc[i].shape
            h3, w3, c3 = test_y_gt[i].shape
            if self.n_channels == 1:
                test_X_bc_ych = test_X_bc[i][:, :, 0]
                test_y_bc_ych = test_y_bc[i][:, :, 0]
                test_y_gt_ych = test_y_gt[i][:, :, 0]
                test_y_bc_cbcr = test_y_bc[i][:, :, 1:3]
                test_X_bc_ych = test_X_bc_ych.reshape([1, h1, w1, 1])
                test_y_bc_ych = test_y_bc_ych.reshape([1, h2, w2, 1])
                test_y_gt_ych = test_y_gt_ych.reshape([1, h3, w3, 1])
                results = self.sess.run(self.result, feed_dict={self.X: test_X_bc_ych, self.y: test_y_gt_ych})
                test_y_bc_ych = test_y_bc_ych[0]
                test_y_gt_ych = test_y_gt_ych[0]
                results = results[0]
                gt = test_y_gt[i]
                bc = test_y_bc[i]
                vdsr = np.concatenate((results, test_y_bc_cbcr), axis=2)
                save_res(self.result_dir, gt, bc, vdsr, i, self.colour_format)
                bicubic_psnr.append(psnr(test_y_bc_ych, test_y_gt_ych))
                test_psnr.append(psnr(results, test_y_gt_ych))
            elif self.n_channels == 3:
                test_X_bc_ = test_X_bc[i]
                test_y_bc_ = test_y_bc[i]
                test_y_gt_ = test_y_gt[i]
                test_X_bc_ = test_X_bc_.reshape([1, h1, w1, c1])
                test_y_bc_ = test_y_bc_.reshape([1, h2, w2, c2])
                test_y_gt_ = test_y_gt_.reshape([1, h3, w3, c3])
                results = self.sess.run(self.result, feed_dict={self.X: test_X_bc_, self.y: test_y_gt_})
                test_y_bc_ = test_y_bc_[0]
                test_y_gt_ = test_y_gt_[0]
                results = results[0]
                gt = test_y_gt[i]
                bc = test_y_bc[i]
                vdsr = results
                save_res(self.result_dir, gt, bc, vdsr, i, self.colour_format)

                bicubic_psnr.append(psnr(test_y_bc_, test_y_gt_))
                test_psnr.append(psnr(results, test_y_gt_))
            else:
                print("Invalid Argument for n_channels")

        for p in range(len(bicubic_psnr)):
            print("Bicubic PSNR of Image {}: {:.2f}".format(p, bicubic_psnr[p]))
        for q in range(len(test_psnr)):
            print("VDSR PSNR of Image {}: {:.2f}".format(q, test_psnr[q]))
        print("Average Bicubic PSNR: {:.2f}".format(np.mean(bicubic_psnr)))
        print("Average VDSR PSNR: {:.2f}".format(np.mean(test_psnr)))
        end_time = time.time()
        print("Time taken: {}".format(end_time - start_time))

    def save(self):
        path = self.model_dir
        if not os.path.exists(path):
            os.mkdir(self.model_dir)
        self.saver.save(self.sess, self.model_dir + self.model_name, global_step=self.epochs)

    def load(self):
        path = self.model_dir
        if path:
            checkpoint_path = tf.train.latest_checkpoint(path)
            self.saver.restore(self.sess, checkpoint_path)
            print("Model Loaded from {}".format(self.model_dir))
        else:
            print("No model to load")


def psnr(x, y):
    mse = np.mean(np.square(np.subtract(x, y)))
    if mse == 0:
        return 100
    else:
        return 10 * np.log10(1. / mse)


def load_matlab_data(train_dir, colour_format):
    if colour_format == 'ych':
        train_dir = train_dir + '/train_91_ychannels_matlab.h5'
    elif colour_format == 'ycbcr':
        train_dir = train_dir + '/train_91_ycbcrchannels_matlab.h5'
    elif colour_format == 'rgb':
        train_dir = train_dir + '/train_91_rgbchannels_matlab.h5'
    with h5py.File(train_dir, 'r') as f:
        x = np.array(f.get('data'))
        y = np.array(f.get('label'))
        return x, y


def load_octave_data(train_dir, colour_format):
    if colour_format == 'ych':
        train_dir = train_dir + '/train_91_ychannels_octave.h5'
    elif colour_format == 'ycbcr':
        train_dir = train_dir + '/train_91_ycbcrchannels_octave.h5'
    elif colour_format == 'rgb':
        train_dir = train_dir + '/train_91_rgbchannels_octave.h5'
    with h5py.File(train_dir, 'r') as f:
        x = np.array(f.get('data').get('value'))
        y = np.array(f.get('label').get('value'))
        return x, y


def imread(path):
    return Image.open(path)


def make_matlab_bc_data(train_dir, scale, colour_format):
    scale = scale
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_mat_ycbcr/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_mat_ycbcr/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_mat_ycbcr/'
        else:
            print("Invalid value for scale")
    elif colour_format == 'rgb':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_mat_rgb/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_mat_rgb/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_mat_rgb/'
        else:
            print("Invalid value for scale")
    dir_list = os.listdir(lr_us_path)
    x = []
    y = []
    count = 0
    for file in dir_list:
        count += 1
        x_ = imread(os.path.join(lr_us_path, file))
        y_ = imread(os.path.join(lr_us_path, file))
        x_ = np.array(x_)
        y_ = np.array(y_)
        x.append(x_ / 255.)
        y.append(y_ / 255.)
    return x, y


def make_octave_bc_data(train_dir, scale, colour_format):
    scale = scale
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_oct_ycbcr/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_oct_ycbcr/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_oct_ycbcr/'
        else:
            print("Invalid value for scale")
    elif colour_format == 'rgb':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_oct_rgb/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_oct_rgb/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_oct_rgb/'
        else:
            print("Invalid value for scale")
    dir_list = os.listdir(lr_us_path)
    x = []
    y = []
    count = 0
    for file in dir_list:
        count += 1
        x_ = imread(os.path.join(lr_us_path, file))
        y_ = imread(os.path.join(lr_us_path, file))
        x_ = np.array(x_)
        y_ = np.array(y_)
        x.append(x_ / 255.)
        y.append(y_ / 255.)
    return x, y


def make_matlab_gt_data(train_dir, colour_format):
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        gt_path = path + '_gt_mat_ycbcr/'
    elif colour_format == 'rgb':
        gt_path = path + '_gt_mat_rgb/'
    dir_list = os.listdir(gt_path)
    y = []
    count = 0
    for file in dir_list:
        count += 1
        y_ = imread(os.path.join(gt_path, file))
        y_ = np.array(y_)
        y.append(y_ / 255.)
    return y


def make_octave_gt_data(train_dir, colour_format):
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        gt_path = path + '_gt_oct_ycbcr/'
    elif colour_format == 'rgb':
        gt_path = path + '_gt_oct_rgb/'
    dir_list = os.listdir(gt_path)
    y = []
    count = 0
    for file in dir_list:
        count += 1
        y_ = imread(os.path.join(gt_path, file))
        y_ = np.array(y_)
        y.append(y_ / 255.)
    return y


def imsave_ycbcr(img, path, filename):
    img = img * 255.
    img = Image.fromarray(img.astype('uint8'), mode='YCbCr')
    img = img.convert('RGB')
    return img.save(os.path.join(path, filename))


def imsave_rgb(img, path, filename):
    img = img * 255.
    img = Image.fromarray(img.astype('uint8'), mode='RGB')
    return img.save(os.path.join(path, filename))


def save_res(path, gt, bc, vdsr, i, colour_format):
    if colour_format == 'ych' or colour_format == 'ycbcr':
        imsave_ycbcr(gt, path, str(i) + '_gt.bmp')
        imsave_ycbcr(bc, path, str(i) + '_bc.bmp')
        imsave_ycbcr(vdsr, path, str(i) + '_vdsr.bmp')
    elif colour_format == 'rgb':
        imsave_rgb(gt, path, str(i) + '_gt.bmp')
        imsave_rgb(bc, path, str(i) + '_bc.bmp')
        imsave_rgb(vdsr, path, str(i) + '_vdsr.bmp')
    else:
        print("Improper value for colour_format")
