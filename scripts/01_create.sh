#!/bin/bash
# Usage: bash scripts/01_create.sh


sudo docker run \
    --name dlcore \
    --gpus all \
    -p 6001:6001 \
    -p 1080:1080 \
    -p 1080:1080/udp \
    -v /media/young/OuterspaceTech./Data/Media:/home/young/Data/Media \
    -v /media/young/OuterspaceTech./Data/Documents:/home/young/Data/Documents \
    -v /media/young/OuterspaceTech./Data/Softwares/Packages:/home/young/Data/Softwares/Packages \
    --privileged=true \
    -dit dlcore:base 
