FROM quay.io/pypa/manylinux2010_x86_64

RUN yum install -y wget libpcap libpcap-devel libnl python-devel libnl-devel valgrind-devel redhat-rpm-config rpm-build gcc

RUN wget https://ufpr.dl.sourceforge.net/project/boost/boost/1.70.0/boost_1_70_0.tar.bz2 -O /tmp/boost_1_70_0.tar.bz2
RUN tar -C /tmp -jxf /tmp/boost_1_70_0.tar.bz2 && \
    cd /tmp/boost_1_70_0 && \
    ./bootstrap.sh --prefix=/opt/boost_1_70_0 --with-libraries=program_options,system && \
    ./b2 cxxflags=-fPIC link=static install && \
    cd / && rm -rf /tmp/boost_1_70_0*

RUN wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-4.5-1.0.1.0/MLNX_OFED_SRC-4.5-1.0.1.0.tgz -O /tmp/MLNX_OFED_SRC-4.5-1.0.1.0.tgz
RUN tar -C /tmp -zxf /tmp/MLNX_OFED_SRC-4.5-1.0.1.0.tgz
RUN cd /tmp/MLNX_OFED_SRC-4.5-1.0.1.0 && \
    echo -e "libibverbs=y\nlibibverbs-devel=y\nlibrdmacm=y\nlibrdmacm-devel=y\n" > ofed.cfg && \
    ./install.pl --user-space-only -c ofed.cfg

RUN cd /opt && \
    git clone https://github.com/luigirizzo/netmap && \
    git -C netmap reset --hard 454ef9c   # A known good version

ENV CPATH=/opt/boost_1_70_0/include:/opt/netmap/sys LIBRARY_PATH=/opt/boost_1_70_0/lib

COPY requirements.txt /tmp/spead2/requirements.txt
COPY manylinux/install_requirements.sh /tmp/spead2/manylinux/install_requirements.sh
RUN /tmp/spead2/manylinux/install_requirements.sh

COPY . /tmp/spead2
RUN /tmp/spead2/manylinux/generate_wheels.sh
