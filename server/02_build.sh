#!/bin/bash
# Usage: bash 02_build.sh

if [[ -f 00_shadowsocks_all.sh ]];then
  chmod +x 00_shadowsocks_all.sh && ./00_shadowsocks_all.sh 2>&1 | tee 00_shadowsocks_all.log
else  
  wget --no-check-certificate -O 00_shadowsocks_all.sh https://raw.githubusercontent.com/yangdu0731/dlcore/base/server/00_shadowsocks_all.sh && chmod +x 00_shadowsocks_all.sh && ./00_shadowsocks_all.sh 2>&1 | tee 00_shadowsocks_all.log
fi

if [[ -f 01_bbr.sh ]];then
  chmod +x 01_bbr.sh && ./01_bbr.sh
else
  wget --no-check-certificate https://raw.githubusercontent.com/yangdu0731/dlcore/base/server/01_bbr.sh && chmod +x 01_bbr.sh && ./01_bbr.sh
fi
