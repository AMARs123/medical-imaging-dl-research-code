#!/bin/bash
# Fetches the two HAM10000 image archives with pdownload.py, then re-checks all six files (download.sh).
P=$HOME/Journal_Papers_Research/Paper_02
cd $P/data/raw
PY=$HOME/Journal_Papers_Research/shared_resources/envs/jpr/bin/python
rm -f HAM10000_images_part_1.zip HAM10000_images_part_2.zip
$PY $P/src/pdownload.py 3172585 HAM10000_images_part_1.zip 1366522108 4639bfa73ab251610530a97c898e6e46 16 &
$PY $P/src/pdownload.py 3172584 HAM10000_images_part_2.zip 1403566547 da43d6cc50f6613013be07e8986b384b 16 &
wait
./download.sh
