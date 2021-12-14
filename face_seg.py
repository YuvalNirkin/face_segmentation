import os
import argparse
import numpy as np
from PIL import Image
import surgery
from pathlib import Path


import caffe


def segment(image_path):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(image_path)
    im = im.resize((500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    # save result to .npy file
    # file_name, file_ext = os.path.splitext(input.split("/"))
    file_name = Path(image_path).stem
    output_file_path = os.path.join(args.output, file_name + '.npy')
    np.save(output_file_path, out)
    print "Done!" , output_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Segmentation')
    parser.add_argument('--input', type=str, default='./input', help='input image or directory path')
    parser.add_argument('--output', type=str, default='./output', help="output directory path")
    args = parser.parse_args()

    print "Process started"

    # if you want to run on gpu, uncomment these 2 lines:
    # caffe.set_device(0)
    # caffe.set_mode_gpu()

    # load net
    net = caffe.Net('data/face_seg_fcn8s_deploy.prototxt', 'data/face_seg_fcn8s.caffemodel', caffe.TEST)

    if os.path.isfile(args.input):
        segment(args.input)

    elif os.path.isdir(args.input):
        for name in os.listdir(args.input):
            image_path = os.path.join(args.input, name)
            if os.path.isfile(image_path):
                segment(image_path)

    print "Process finished"
