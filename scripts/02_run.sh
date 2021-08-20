#!/bin/bash
# Usage: bash scripts/02_run.sh

sudo docker start dlcore
sudo docker exec -it dlcore bash
