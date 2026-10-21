#!/bin/bash
# Extract the selected NIH ChestX-ray14 images from the Hugging Face mirror (verified identical to the official NIH
# archive for all 4,999 images of official part 1; see logs/verify_nih_mirror.log), CRC-32 checked per file.
P=$HOME/Journal_Papers_Research/Paper_04
PY=$HOME/Journal_Papers_Research/shared_resources/envs/jpr/bin/python
cd $P/data/raw
for i in 001 002 003 004 005 006 007 008 009 010 011 012; do
  $PY ../../../shared_resources/remote_zip.py extract https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_$i.zip nih nih_wanted.txt 24
done
echo NIH_FETCH_DONE
