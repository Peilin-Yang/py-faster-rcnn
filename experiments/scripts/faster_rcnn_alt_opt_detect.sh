#!/bin/bash
#output/faster_rcnn_end2end/bib_500X500Gray_training/vgg16_faster_rcnn_iter_70000.caffemodel \
time ./tools/detect_net.py --gpu 0 \
  --def models/bib/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net $1 \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  --input $2 \
  --output $3
  