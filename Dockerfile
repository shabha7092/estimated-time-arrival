FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv python3.6-tk
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
CMD python3.6 -m unittest discover ; python3.6 analysis.py
