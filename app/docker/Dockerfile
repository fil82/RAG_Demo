FROM python:3.10 AS requirements-stage

WORKDIR /tmp

RUN pip install poetry==1.5.1

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10

WORKDIR /app

COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
ENV WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}
COPY . /app/

COPY ./pyproject.toml /app/
COPY git-manifest.yaml /app/git-manifest.yaml
COPY logging.yaml /app/logging.yaml
RUN mkdir -p /app/logs && chmod 0777 /app/logs

RUN chmod 777 -R /app

# Heroku uses PORT, Azure App Services uses WEBSITES_PORT, Fly.io uses 8080 by default
CMD exec uvicorn server.main:app --host 0.0.0.0 --workers ${WEB_CONCURRENCY} --port ${PORT:-${WEBSITES_PORT:-8080}}
