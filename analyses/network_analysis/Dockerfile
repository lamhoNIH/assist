FROM rocker/r-ver:4.0.3

COPY install_packages.R /install_packages.R
COPY wgcna_codes.R /wgcna_codes.R
COPY version.txt /version.txt
WORKDIR /

RUN Rscript install_packages.R

CMD Rscript wgcna_codes.R