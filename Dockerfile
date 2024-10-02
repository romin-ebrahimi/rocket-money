FROM python:3.9.13

COPY . .
WORKDIR /

RUN pip3 install --upgrade pip && \
    pip3 install \
        --root-user-action=ignore \
        --no-cache-dir \
        --upgrade \
        -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "model_bert.py"]