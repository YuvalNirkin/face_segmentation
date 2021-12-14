# FROM bvlc/caffe:gpu
FROM bvlc/caffe:cpu


RUN apt-get -y update
RUN apt-get -y install python-tk
RUN pip install pathlib

WORKDIR /workspace/face_segmentation

ENTRYPOINT [ "python" , "face_seg.py" ]