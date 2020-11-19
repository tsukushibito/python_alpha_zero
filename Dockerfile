FROM python:3

RUN apt-get update
RUN apt-get  install -y vim less graphviz default-jre
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade pipenv