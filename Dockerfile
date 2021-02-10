FROM continuumio/miniconda3 AS gazetteer

RUN apt-get update
RUN apt-get -qy install vim
RUN apt-get -qy install build-essential

COPY GAZ_group.csv /home/COVID_Gazetteer/
COPY requirements.txt /home/COVID_Gazetteer/
ADD scripts/ /home/COVID_Gazetteer/
WORKDIR /home/COVID_Gazetteer/

RUN pip install -r requirements.txt
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz 
RUN python -m spacy download en_core_web_sm
