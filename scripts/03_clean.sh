#!/bin/bash
# Usage: bash scripts/03_clean.sh


sudo docker stop dlcore
sudo docker rm dlcore
sudo docker image rm dlcore:base
#sudo docker image rm ubuntu:20.04
