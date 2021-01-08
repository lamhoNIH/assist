FROM rocker/r-ver:4.0.3

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev python3.8 python3-pip python3-setuptools python3-dev
RUN pip3 install --upgrade pip

RUN mkdir /assist

COPY requirements.txt /assist
COPY requirements.r /assist
COPY src /assist/src
COPY preproc.py /assist
COPY wgcna_codes.R /assist
COPY version.txt /assist
COPY run.sh /assist
RUN chmod +x /assist/run.sh
WORKDIR /assist

RUN pip3 install -r /assist/requirements.txt
RUN Rscript /assist/requirements.r

CMD ["./run.sh"]