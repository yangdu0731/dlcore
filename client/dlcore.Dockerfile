from ubuntu:20.04

# Arguments
arg user_name
arg user_uid
arg user_gid
arg user_group
arg user_dir
arg softwares_dir

arg server
arg server_port
arg local_address
arg local_port
arg password
arg timeout
arg method

# System
run apt-get update
run apt-get install -y systemctl
run apt-get install -y shadowsocks-libev && \
echo "{\"server\":\""$server"\",\"server_port\":"$server_port",\"local_address\":\""$local_address"\",\"local_port\":"$local_port",\"password\":\"$password\",\"timeout\":"$timeout",\"method\":\""$method"\"}" >> /etc/shadowsocks-libev/local.json && \
sed -i "s/%i/local/g" /lib/systemd/system/shadowsocks-libev-local@.service
run apt-get install -y python3-pip
run pip install genpac
run pip install --upgrade genpac
run apt-get install -y vim
run apt-get install -y wget
run apt-get install -y git
run apt-get install -y libgl1-mesa-glx
run apt-get install -y sudo

# User
run groupadd -r -g $user_gid $user_name && \
useradd --no-log-init -m -r -g $user_group -u $user_uid $user_name -s /bin/bash -p $user_name && \
mkdir -p $user_dir && \
chown -fR $user_name:$user_name $user_dir && \
chmod 777 /etc/sudoers && \
echo "$user_name ALL=(ALL:ALL) ALL" >> /etc/sudoers && \
chmod 555 /etc/sudoers && \
echo "$user_name\n$user_name\n" |passwd $user_name
user $user_name
workdir $user_dir
run echo "PS1='\e[1;37m[\e[m\e[1;32m\u\e[m\e[1;33m@\e[m\e[m\e[1;35mdlcore\e[m \e[4m\`pwd\`\e[m\e[1;37m]\e[m\e[1;36m\e[m\n$'" >> ~/.bashrc && \
echo "alias ll='ls -alFhX'" >> ~/.bashrc && \
echo "echo "$user_name" |sudo -S systemctl start shadowsocks-libev-local@." >> ~/.bashrc && \
echo 'genpac --pac-proxy "SOCKS5 127.0.0.1:1080" --gfwlist-proxy="SOCKS5 127.0.0.1:1080" --gfwlist-url=https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt --output="~/autoproxy.pac"' >> ~/.bashrc && \
echo "clear" >> ~/.bashrc
run echo "set nu\nset ts=4\nset expandtab\nset autoindent" >> ~/.vimrc

# Softwares
run mkdir -p $softwares_dir/Packages && mkdir -p $softwares_dir/Installations
## Anaconda3
add Anaconda*.sh $softwares_dir/Packages
run echo "\nyes\n$softwares_dir/Installations/anaconda3\nyes\n" |bash $softwares_dir/Packages/Anaconda*.sh
run bash -c "source ~/.bashrc" 
## PyTorch1.4.0
run $softwares_dir/Installations/anaconda3/bin/conda create \
    -n algorithm \
    tensorflow-gpu=2.2.0 \
    numpy=1.16.6 \
    tensorboard=2.6.0 \
    pandas=1.2.3 \
    opencv=4.2.0 \
    cudatoolkit=10.1 \
    pyyaml=5.3.1 \
    python=3.8 \
    pytorch=1.7.0 \
    torchvision=0.8.1 \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    -c anaconda
run echo "conda activate algorithm" >> ~/.bashrc
## Clean
run rm -rf $softwares_dir/Packages/*

# CMD
cmd ["/bin/bash", "-c", "tail -f /dev/null"]
