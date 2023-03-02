FROM python:3.8.12

COPY simplyJapanese /simplyJapanese
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn simplyJapanese.api.fast:app --host 0.0.0.0
