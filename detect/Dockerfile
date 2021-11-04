# FROM python:3.7-slim-buster
FROM tensorflow/tensorflow:2.6.0-gpu

WORKDIR /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root:/root/workflows

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --yes python-opencv

# Install Python dependencies
COPY ./requirements.txt /root
RUN pip install -r /root/requirements.txt
RUN pip install backports-datetime-fromisoformat
# Copy the actual code
COPY ./run_luigi.sh /root
COPY ./workflows /root/workflows
COPY ./scripts /root/scripts
COPY ./luigi.cfg /root

ENTRYPOINT ["/root/run_luigi.sh"]