FROM nersc/tensorflow:ngc-20.09-tf2-v0
WORKDIR /app

# set up software directory outside of /root for shifter
RUN mkdir -p /software/lib/python3.6/site-packages 
ENV PIP_TARGET /software/lib/python3.6/site-packages
ENV PYTHONPATH /software/lib/python3.6/site-packages:$PYTHONPATH

# get dependencies for namaster
RUN apt-get update && apt-get install -y \
        automake                         \
        build-essential                  \
        gfortran                         \
        libcfitsio-dev                   \
        libfftw3-dev                     \
        libgsl-dev                       \
        libchealpix-dev            

# install various python packages
RUN python -m pip install pip numpy ipython jupyter
RUN python -m pip install --upgrade astropy healpy toml
RUN python -m pip install --upgrade cosmoplotian
RUN python -m pip install --upgrade pymaster
RUN python -m pip install --upgrade gin-config absl-py
RUN python -m pip install --upgrade tensorflow-probability==0.11