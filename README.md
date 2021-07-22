# VDSR_TensorFlow

## About

This is the implementation of VDSR Paper using TensorFlow.

## Requirements

TensorFlow 1.15.5 \
Pillow (PIL) \
Numpy \
h5py \
MATLAB or GNU Octave (For Data Preprocessing)

## Implementation Details

In this implementation, the VDSR network can be trained on both 1 channel and 3 channels. The data pre-processing can be done using either MATLAB or GNU Octave (Link to files: https://github.com/gokhaleshreyas/Data_Generation_Files).

Note: The data generation files for this specific section needs to be downloaded and kept in the 'data_gen' folder.

## Data Augmentation

As per mentioned in the paper, the data augmentation is done in the following way:
1. Image Rotation: The sub-images and sub-labels are rotated by the interval 90 degrees creating 4 sub-images and sub-labels instead of 1.
2. Image Flipping: The sub-images and sub-labels are flipped using the image fliplr() function and then, these flipped sub-images and sub-labels are also rotated by 90 degrees.

## Key Features

1. The learning rate decay is used. The only change is the initial learning rate is kept at 0.01 instead of 0.1 in the paper.
2. Provision for Gradient Clipping

## Use

### Data Pre-processing

The MATLAB as well as GNU Octave Code is given for data pre-processing. The training data will be generated in hdf5 format.
As mentioned in the paper, the author's have used the 291 image dataset. But here, the 91 image dataset is used (mentioned by the authors in the same paper).

Steps:
1. Download the 291 image dataset from the author's website (Link is in the reference section) and put them in the 'Dataset' directory.
2. Run the appropriate code for train data generation. For example, to create training data for y-channel using GNU Octave, run the 'data_gen_h5_octave_ychannel.m' file in GNU Octave to get the 'train_291_ychannels_octave.h5' file for training in the train directory.
3. Run the appropriate code for test data generation. For example, to create test data for y-channel, run the 'data_gen_octave_rgb2ycbcr.m' file to get the test data in test directory.

Note: Both the train file and test images are needed to be generated before initiating the training.

### Training

The training requires GPU. To train the network, follow the following steps:
1. Open terminal in the SRCNN_TensorFlow folder.
2. type 'python main.py' and select the appropriate arguments to begin the training.
3. Set the '--do_train' value to True to start training the network.


### Testing

To test the trained network, ensure that the trained weights are in the model directory. Follow the same steps as done in training, except, set the '--do_test' value to True to test the network performance.


### Arguments (To be changed while training)
The arguments are: \
-h, --help            show this help message and exit \
--do_train:  To Start training, default value: False \
--do_test: To Start testing, default value: False \
--train_dir: Enter a different training directory than default \
--valid_dir: Enter a different validation directory than default \
--test_dir: Enter a different testing directory than default \
--model_dir: Enter a different model directory than default \
--result_dir: Enter a different result directory than default \
--scale: Enter a scale value among 2,3,4, default value: 3 \
--learning_rate: Enter learning rate, default value: 1e-4 \
--momentum: Enter a momentum value for SGD Optimizer, default value: 0.9 \
--epochs: Enter the number of epochs, default value: 1000 \
--n_channels: Enter number of channels (1(y-channel) or 3(rgb or ycbcr)), default value: 1 \
--batch_size: Enter the batch size, default value: 128 \
--colour_format: Enter the colour format among ych, ycbcr, rgb, default value: ych \
--depth: The depth of the network, default value: 20 \
--prepare_data: Enter the string for data preparation (used in data pre-processing), default value: 'matlab'



## References

- [Official Website][1]
    - Reference to the original code.

- [jinsuyoo/vdsr][2]
    - Reference to the referred repository.

- [gokhaleshreyas/srcnn][3]
    - Reference to the other repository.

[1]: https://cv.snu.ac.kr/research/VDSR/
[2]: https://github.com/jinsuyoo/srcnn
[3]: https://github.com/gokhaleshreyas/SRCNN_TensorFlow
