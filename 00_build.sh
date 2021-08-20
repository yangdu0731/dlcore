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
server="108.61.186.100" # TODO
server_password='k1)X.5VB.31p!17r' # TODO
server_port="9269"
shadowsocks_password="d15882706805"
timeout="60"
method="aes-256-gcm"

# client
sudo docker build \
    --build-arg server=$server \
    --build-arg server_port=$server_port \
    --build-arg shadowsocks_password=$shadowsocks_password \
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

# server    
ssh-keygen -f "/home/$user_name/.ssh/known_hosts" -R $server
sshpass -p $server_password scp -o "StrictHostKeyChecking no" utils/shadowsocks*.sh root@$server:/root
sshpass -p $server_password ssh -o "StrictHostKeyChecking no" root@$server bash shadowsocks_run.sh $shadowsocks_password $server_port $method
