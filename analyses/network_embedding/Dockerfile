FROM continuumio/miniconda3

RUN mkdir /assist

COPY requirements.txt /assist
COPY network_embedding.py /assist
COPY src/preproc /assist/preproc
COPY src/eda /assist/eda
COPY src/embedding /assist/embedding
COPY version.txt /assist
COPY run.sh /assist
RUN chmod +x /assist/run.sh
WORKDIR /assist

RUN pip install -r /assist/requirements.txt
#RUN pip uninstall -y csrgraph
#RUN git clone -b NET-1750-reduce-memory https://github.com/netrias/CSRGraph.git
#RUN pip install --upgrade --no-cache-dir git+https://github.com/netrias/CSRGraph.git@NET-1750-reduce-memory
###RUN pip install --upgrade --no-cache-dir git+https://github.com/netrias/CSRGraph.git
#RUN pip install -e CSRGraph
CMD ["./run.sh"]