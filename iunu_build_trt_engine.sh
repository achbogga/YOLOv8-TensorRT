#!/bin/bash
image_size=640
weights_prefix="/home/aboggaram/models/Octiva/octiva_best_yolov8_models_desktop/octiva_yolov8_seg_2023_01_17_imgsz_640_best"
pytorch_weights="${weights_prefix}.pt"
onnx_weights="${weights_prefix}.onnx"
python3 export-seg.py \
--weights "${pytorch_weights}" \
--opset 11 \
--sim \
--input-shape 1 3 ${image_size} ${image_size} \
--device cuda:0

python3 build.py \
--weights "${onnx_weights}" \
--fp16  \
--device cuda:0 \
--seg