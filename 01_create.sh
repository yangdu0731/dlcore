#!/bin/bash
# Usage: bash create.sh


sudo docker run \
    --name dlcore \
    --gpus all \
    -p 6001:6001 \
    -v /media/young/OuterspaceTech./Data/Media:/home/young/Data/Media \
    --privileged=true \
    -dit dlcore:base 
