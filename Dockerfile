FROM python:3.9

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN python3 -m venv /opt/venv

COPY requirements.txt /app/requirements.txt

RUN /opt/venv/bin/pip install -r requirements.txt

COPY . .

RUN /opt/venv/bin/pip install pip --upgrade && \
    chmod +x entrypoint.sh

EXPOSE 1111

CMD ["/app/entrypoint.sh"]