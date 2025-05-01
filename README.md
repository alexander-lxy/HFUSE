# HFUSE
查看cuda是源码还是apt安装
dpkg -l | grep cuda

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/bin/activate

conda create -n env1 python=3.9
# 关闭自启动 conda config --set auto_activate_base false

conda activate env1

srun --gpus-per-node=1 --partition=gpu-preempt --pty /bin/bash -i

pip install torch --pre
#可能提示缺少 pip install numpy

python setup.py install build_ext -j8

python test.py