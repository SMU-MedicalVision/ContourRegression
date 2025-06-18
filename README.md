# Difficulty-aware Coupled Contour regression network

This repository contains the original models and code described in the paper "Difficulty-aware Coupled Contour Regression Network with IoU Loss for Efficient IVUS Delineation".

## Deployment

See the requirements.txt to construct the processing environment.

## Training

Run

'''
python train.py
'''

The input IVUS images are consecutive frames.

## Evaluation

'''
python evluation.py
'''

## Inference

For any random IVUS images as the model input, run

'''
python inference.py
'''

to obtain the lumen and media contours.
