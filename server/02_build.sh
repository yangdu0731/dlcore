#!/bin/bash
# Usage: bash 02_build.sh


wget --no-check-certificate -O 00_shadowsocks_all.sh https://raw.githubusercontent.com/yangdu0731/dlcore/base/server/00_shadowsocks_all.sh && chmod +x 00_shadowsocks_all.sh && ./00_shadowsocks_all.sh 2>&1 | tee 00_shadowsocks_all.log
wget --no-check-certificate https://raw.githubusercontent.com/yangdu0731/dlcore/base/server/01_bbr.sh && chmod +x 01_bbr.sh && ./01_bbr.sh
