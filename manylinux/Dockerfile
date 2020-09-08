FROM quay.io/pypa/manylinux2010_x86_64

RUN yum install -y \
        wget libpcap libpcap-devel libnl python-devel libnl-devel \
        valgrind-devel redhat-rpm-config rpm-build \
        cmake3 ninja-build pandoc libnl3-devel

RUN wget https://ufpr.dl.sourceforge.net/project/boost/boost/1.74.0/boost_1_74_0.tar.bz2 -O /tmp/boost_1_74_0.tar.bz2
RUN tar -C /tmp -jxf /tmp/boost_1_74_0.tar.bz2 && \
    cd /tmp/boost_1_74_0 && \
    ./bootstrap.sh --prefix=/opt/boost_1_74_0 --with-libraries=program_options,system && \
    ./b2 cxxflags=-fPIC link=static install && \
    cd / && rm -rf /tmp/boost_1_74_0*

RUN wget https://github.com/linux-rdma/rdma-core/releases/download/v31.0/rdma-core-31.0.tar.gz -O /tmp/rdma-core-31.0.tar.gz
RUN tar -C /tmp -zxf /tmp/rdma-core-31.0.tar.gz
RUN cd /tmp/rdma-core-31.0 && ./build.sh
# See https://github.com/pypa/manylinux/issues/731
RUN cp /usr/share/aclocal/pkg.m4 /usr/local/share/aclocal/

ENV CPATH=/opt/boost_1_74_0/include LIBRARY_PATH=/opt/boost_1_74_0/lib PKG_CONFIG_PATH=/tmp/rdma-core-31.0/build/lib/pkgconfig

COPY requirements.txt /tmp/spead2/requirements.txt
COPY manylinux/install_requirements.sh /tmp/spead2/manylinux/install_requirements.sh
RUN /tmp/spead2/manylinux/install_requirements.sh

COPY . /tmp/spead2
RUN /tmp/spead2/manylinux/generate_wheels.sh
