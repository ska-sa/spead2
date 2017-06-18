FROM quay.io/pypa/manylinux1_x86_64

RUN curl https://ufpr.dl.sourceforge.net/project/boost/boost/1.64.0/boost_1_64_0.tar.bz2 > /tmp/boost_1_64_0.tar.bz2
RUN tar -C /tmp -jxf /tmp/boost_1_64_0.tar.bz2 && \
    cd /tmp/boost_1_64_0 && \
    ./bootstrap.sh --prefix=/opt/boost_1_64_0 --with-libraries=program_options,system && \
    ./b2 cxxflags=-fPIC link=static install && \
    cd / && rm -rf /tmp/boost_1_64_0*

ENV CPATH=/opt/boost_1_64_0/include LIBRARY_PATH=/opt/boost_1_64_0/lib

COPY requirements.txt /tmp/spead2/requirements.txt
COPY manylinux1/install_requirements.sh /tmp/spead2/manylinux1/install_requirements.sh
RUN /tmp/spead2/manylinux1/install_requirements.sh

COPY . /tmp/spead2
RUN /tmp/spead2/manylinux1/generate_wheels.sh
