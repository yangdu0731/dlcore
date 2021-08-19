#!/bin/bash
# Usage: bash build.sh

# env
user_name="young"
id_info=$(id |grep uid)
id_info=(${id_info// / })
user_uid=${id_info[0]#*=}
user_uid=${user_uid%%(*}
user_gid=${id_info[1]#*=}
user_gid=${user_gid%%(*}
user_group=${id_info[2]#*=}
user_group=${user_group%%(*}
user_dir="/home/$user_name/Data"
softwares_dir="$user_dir/Softwares"

# web
server="141.164.52.122"
server_port="9269"
local_address="0.0.0.0"
local_port="1080"
password="d15882706805"
timeout="60"
method="aes-256-gcm"

sudo docker build \
    --build-arg server=$server \
    --build-arg server_port=$server_port \
    --build-arg local_address=$local_address \
    --build-arg local_port=$local_port \
    --build-arg password=$password \
    --build-arg timeout=$timeout \
    --build-arg method=$method \
    --build-arg user_name=$user_name \
    --build-arg user_uid=$user_uid \
    --build-arg user_gid=$user_gid \
    --build-arg user_group=$user_group \
    --build-arg user_dir=$user_dir \
    --build-arg softwares_dir=$softwares_dir \
    -f dlcore.Dockerfile \
    -t dlcore:base \
    $softwares_dir/Packages
