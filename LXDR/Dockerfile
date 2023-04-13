FROM ubuntu:18.04
FROM python:3.7.8
    
RUN apt-get update
RUN apt-get dist-upgrade -y

WORKDIR /usr/src/app

COPY req.txt .

RUN pip install --upgrade setuptools
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r req.txt
RUN python3 -m pip install --upgrade notebook

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]