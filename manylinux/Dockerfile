FROM quay.io/pypa/manylinux2014_x86_64

RUN yum install -y \
        wget libpcap libpcap-devel python-devel \
        cmake3 ninja-build pandoc libnl3-devel

RUN wget https://ufpr.dl.sourceforge.net/project/boost/boost/1.74.0/boost_1_74_0.tar.bz2 -O /tmp/boost_1_74_0.tar.bz2
RUN tar -C /tmp -jxf /tmp/boost_1_74_0.tar.bz2 && \
    cd /tmp/boost_1_74_0 && \
    ./bootstrap.sh --prefix=/opt/boost_1_74_0 --with-libraries=program_options,system && \
    ./b2 cxxflags=-fPIC link=static install && \
    cd / && rm -rf /tmp/boost_1_74_0*

RUN wget https://github.com/linux-rdma/rdma-core/releases/download/v31.0/rdma-core-31.0.tar.gz -O /tmp/rdma-core-31.0.tar.gz
RUN tar -C /tmp -zxf /tmp/rdma-core-31.0.tar.gz && \
    cd /tmp/rdma-core-31.0 && \
    mkdir build && \
    cd build && \
    cmake3 -GNinja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja-build -v install && \
    cd / && rm -r /tmp/rdma-core-31.0
# See https://github.com/pypa/manylinux/issues/731
RUN cp /usr/share/aclocal/pkg.m4 /usr/local/share/aclocal/

ENV CPATH=/opt/boost_1_74_0/include LIBRARY_PATH=/opt/boost_1_74_0/lib

COPY . /tmp/spead2
RUN /tmp/spead2/manylinux/generate_wheels.sh
