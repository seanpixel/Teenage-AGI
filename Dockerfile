FROM python:3.11-slim

# Set build argument
ARG API_ENABLED

# Set environment variable based on the build argument
ENV API_ENABLED=${API_ENABLED} \
    PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"
RUN pip install poetry

WORKDIR /app
#COPY requirements.txt /tmp/requirements.txt
#RUN pip install -r requirements.txt
COPY pyproject.toml poetry.lock /app/

# Install the dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev

RUN playwright install

WORKDIR /app
COPY . /app
ENTRYPOINT ["python", "main.py"]
