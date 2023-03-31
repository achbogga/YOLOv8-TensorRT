#!/bin/bash
image_size=640
weights_prefix="/home/aboggaram/models/Octiva/octiva_best_yolov8_models_desktop/octiva_yolov8_seg_2023_01_17_imgsz_640_best"
trt_weights="${weights_prefix}.engine"
input_folder="/home/aboggaram/data/Octiva/trt_inference_test_images"
output_folder="/home/aboggaram/data/Octiva/trt_inference_test_images_output"
python3 infer-seg.py \
--engine "${trt_weights}" \
--imgs "${input_folder}" \
--show \
--out-dir "${output_folder}" \
--device cuda:0