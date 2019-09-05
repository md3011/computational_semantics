#!/bin/bash
#$ -cwd
#$ -l inf
#$ gpus=1
#$ -o output_folder/
#$ -e error_folder/
source /course/cs2952d/pytorch-gpu/bin/activate
python3 sentiment.py --device cuda --embedding glove
