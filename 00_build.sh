#!/bin/bash
# Usage: bash build.sh


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

sudo docker build \
    --build-arg user_name=$user_name \
    --build-arg user_uid=$user_uid \
    --build-arg user_gid=$user_gid \
    --build-arg user_group=$user_group \
    --build-arg user_dir=$user_dir \
    --build-arg softwares_dir=$softwares_dir \
    -f dlcore.Dockerfile \
    -t dlcore:base \
    $softwares_dir/Packages
