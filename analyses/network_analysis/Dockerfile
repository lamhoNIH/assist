FROM rocker/r-ver:4.0.3

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev
RUN mkdir /assist

COPY requirements.r /assist
COPY wgcna_codes.R /assist
COPY version.txt /assist
COPY run.sh /assist
RUN chmod +x /assist/run.sh
WORKDIR /assist

RUN Rscript /assist/requirements.r

CMD ["./run.sh"]