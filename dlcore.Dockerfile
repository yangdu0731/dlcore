from ubuntu:20.04

# Arguments
arg user_name
arg user_uid
arg user_gid
arg user_group
arg user_dir
arg softwares_dir

# System
run apt-get update
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
echo "alias ll='ls -alFhX'" >> ~/.bashrc
run echo "set nu\nset ts=4\nset expandtab\nset autoindent" >> ~/.vimrc

# Softwares
run mkdir -p $softwares_dir/Packages && mkdir -p $softwares_dir/Installations
## Anaconda3
add Anaconda*.sh $softwares_dir/Packages
run echo "\nyes\n$softwares_dir/Installations/anaconda3\nyes\n" |bash $softwares_dir/Packages/Anaconda*.sh
run bash -c "source ~/.bashrc" 
run echo "conda deactivate" >> ~/.bashrc
## PyTorch1.4.0
run $softwares_dir/Installations/anaconda3/bin/conda create \
    -n algorithm \
    tensorflow=1.15.0 \
    numpy=1.16.6 \
    tensorboard=1.15.0 \
    pandas=1.2.3 \
    opencv=4.2.0 \
    cudatoolkit=10.1 \
    pyyaml=5.3.1 \
    python=3.7 \
    pytorch=1.8.0 \
    torchvision=0.2.2 \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    -c anaconda
## Clean
run rm -rf $softwares_dir/Packages/*

# CMD
cmd ["/bin/bash", "-c", "tail -f /dev/null"]
