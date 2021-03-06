#!/bin/bash
#output/faster_rcnn_end2end/bib_500X500Gray_training/vgg16_faster_rcnn_iter_70000.caffemodel \
time ./tools/detect_net.py --gpu 0 \
  --def models/bib/VGG16/faster_rcnn_end2end/test.prototxt \
  --net $1 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --input $2 \
  --output $3 \
  --excludes $4
  