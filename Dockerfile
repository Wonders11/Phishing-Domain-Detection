# Dockerfile is a file with set of instructions that can convert application to Docker image
# Docker image to Docker container

# fetching image from docker hub
FROM python:3.8-slim-buster

# setting working directory
WORKDIR /service

# pushing code from local to above directory and . means current directory
COPY requirements.txt .

# copy everything(.) from current directory(./)
COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3","app.py" ]