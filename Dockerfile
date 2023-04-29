FROM python:3.11-slim

# Set build argument
ARG API_ENABLED

# Set environment variable based on the build argument
ENV API_ENABLED=${API_ENABLED} \
    PIP_NO_CACHE_DIR=true

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
#RUN pip install -r requirements.txt
COPY pyproject.toml poetry.lock /app/

# Install the dependencies
RUN poetry install --no-interaction --no-ansi

RUN playwright install

WORKDIR /app
COPY . /app
ENTRYPOINT ["python", "main.py"]
