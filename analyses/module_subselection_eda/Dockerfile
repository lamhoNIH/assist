FROM continuumio/miniconda3

RUN mkdir /assist

COPY requirements.txt /assist
COPY module_subselection_eda.py /assist
COPY src /assist/src
COPY version.txt /assist
COPY run.sh /assist
RUN chmod +x /assist/run.sh
WORKDIR /assist

RUN pip install -r /assist/requirements.txt

CMD ["./run.sh"]