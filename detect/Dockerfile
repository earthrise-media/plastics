FROM python:3.7-slim-buster

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root:/root/workflows

RUN apt-get update && apt-get install -y build-essential
ENV VENV /opt/venv
# Virtual environment
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY ./requirements.txt /root
RUN pip install -r /root/requirements.txt

# Copy the actual code
COPY ./run_luigi.sh /root
COPY ./workflows /root/workflows
COPY ./scripts /root/scripts

ENTRYPOINT /root/run_luigi.sh