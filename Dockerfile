FROM python:3

# Move files to container
COPY src/edge_learn/requirements.txt /edge_learn/requirements.txt

# Install dependencies
RUN pip install -r /edge_learn/requirements.txt

# Add edge_learn & decentralizepy to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/"

WORKDIR /edge_learn/