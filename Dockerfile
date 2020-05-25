FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN /usr/bin/python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN /usr/bin/python3 -m pip install -r requirements.txt
RUN apt-get install -y graphviz
COPY credentials.json /creds/gcp.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/creds/gcp.json" 
