FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Jan Deriu <deri@zahw.ch>

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=0.8.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=1.2.2
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest
ARG CAFFE_VERSION=master

#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#	cat /proc/driver/nvidia/version && \
#	echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#	nvcc -V && \
#	echo -e "\n\nBuilding your Deep Learning Docker Image...\n"

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python3-dev \
		python3-pip \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Add SNI support to Python
RUN pip3 --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python3-numpy \
		python3-scipy \
		python3-nose \
		python3-h5py \
		python3-skimage \
		python3-matplotlib \
		python3-pandas \
		python3-sklearn \
		python3-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install --upgrade ipython && \
	pip3 --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python3 -m ipykernel.kernelspec

RUN pip3 --no-cache-dir install --upgrade nltk
RUN pip3 --no-cache-dir install --upgrade tqdm
RUN pip3 --no-cache-dir install --upgrade gensim 


# Install TensorFlow
RUN pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
# Install Keras
RUN pip3 --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

# Install Theano and set up Theano config (.theanorc) for CUDA and OpenBLAS
RUN pip3 --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc

RUN /bin/sh -c python3 -m nltk.downloader punkt
RUN /bin/sh -c python3 -m nltk.downloader stopwords

# Install Keras
RUN pip3 --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

RUN mkdir /DLFramework

COPY architectures /DLFramework/architectures 
COPY custom_keras_layers /DLFramework/custom_keras_layers
COPY data_loader /DLFramework/data_loader
COPY scripts /DLFramework/scripts
COPY utils /DLFramework/utils
COPY evaluation /DLFramework/evaluation
COPY create_word_embeddings.py /DLFramework
COPY distant_phase_nnet.py /DLFramework
COPY embeddings.py /DLFramework
COPY embeddings_container.py /DLFramework
COPY evaluation_metrics /DLFramework/evaluation_metrics

WORKDIR /DLFramework

CMD python3 -m tensorflow.tensorboard --logdir=logging/distant_it
