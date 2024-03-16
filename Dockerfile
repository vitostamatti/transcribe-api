FROM python:3.12-slim as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.12-slim
RUN apt-get update && apt-get install git -y
RUN apt-get install build-essential -y
RUN apt-get install -y ffmpeg 

WORKDIR /code

COPY . /app

COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 9000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "0", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]