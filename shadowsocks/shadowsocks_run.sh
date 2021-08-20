#!/bin/bash
# Usage: bash shadowsocks_run.sh


password=$1
server_port=$2
method=$3

if [[ $method == "aes-256-gcm" ]];then
  method=1
fi

chmod +x shadowsocks_all.sh
echo -e "4\n$password\n$server_port\n$method\nn\n" |sudo ./shadowsocks_all.sh 2>&1 |tee shadowsocks_all.log

# Speed up
#chmod +x shadowsocks_bbr.sh
#echo -e "y\n" |sudo ./shadowsocks_bbr.sh
