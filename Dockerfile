FROM python:3.11-slim

# Set build argument
ARG API_ENABLED

# Set environment variable based on the build argument
ENV API_ENABLED=${API_ENABLED} \
    PIP_NO_CACHE_DIR=true

WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app
ENTRYPOINT ["python", "main.py"]
