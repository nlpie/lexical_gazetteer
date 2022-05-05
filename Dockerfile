FROM continuumio/miniconda3 AS gazetteer

RUN apt-get update
RUN apt-get -qy install vim
RUN apt-get -qy install build-essential

COPY GAZ_group.csv /home/gazetteer/
COPY requirements.txt /home/gazetteer/
ADD scripts/ /home/gazetteer/
WORKDIR /home/gazetteer/

RUN pip install -r requirements.txt
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz 
RUN python -m spacy download en_core_web_sm

WORKDIR /data
