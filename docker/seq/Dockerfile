FROM broadinstitute/gatk

RUN apt-get update -y && \
    apt-get install -y git && \
    apt-get install -y bcftools

RUN apt-get install libcurl3-dev
RUN apt-get install -y libssl-dev

# HtsLib
RUN wget https://github.com/samtools/htslib/releases/download/1.10.2/htslib-1.10.2.tar.bz2 -O htslib.tar.bz2 && \
    tar -xjvf htslib.tar.bz2 && cd htslib-1.10.2 && ./configure && make && make install

ENV LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}

# HapCut2
RUN git clone https://github.com/vibansal/HapCUT2.git
RUN cd HapCUT2 && export C_INCLUDE_PATH=/usr/include/htslib && make &&  make install