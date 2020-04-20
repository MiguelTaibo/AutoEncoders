#!/bin/bash

# Compute facial mask based on HOG-computed landmarks

PICTURES="$1"
EXPORT="$2"

mkdir "$EXPORT"

for picture in "$PICTURES"*
do
	export_path="landmark_masks/$picture_mask.png"
	echo "> Processing picture $picture"
	python detect_face_parts.py -p shape_predictor_68_face_landmarks.dat -i "$picture" -o "$EXPORT"
done