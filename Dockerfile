FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install pydot
RUN /usr/bin/python3 -m pip install git+git://github.com/jarotter/itam-dl-proyecto.git
RUN apt-get install -y graphviz
COPY credentials.json /creds/gcp.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/creds/gcp.json" 
