# TODO: check that it is working.
FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36
USER root

RUN mkdir ./workspace
WORKDIR ./workspace

RUN apt-get update --fix-missing && apt-get upgrade -y

# installing dependencies
RUN pip uninstall --yes opencv
RUN pip install --upgrade --force-reinstall opencv-python

RUN pip install \
    torchvision \
    torchinfo \
    torcheval \
    mujoco==3.2.0 \
    gymnasium \
    shimmy \
    dm_control==1.0.2 \
    imageio \
    imageio-ffmpeg \
    Pillow \
    pyrallis \
    h5py \
    tqdm \
    stable_baselines3 \
    sb3-contrib \
    vector-quantize-pytorch \
    wandb==0.19.9

USER user