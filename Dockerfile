# Use an official Python runtime as a parent image
FROM python:3.6-slim

WORKDIR /FraudDetection

ADD . /FraudDetection

RUN pip install -r requirements.txt

ENV NAME FraudDetection

CMD ["python", "main.py"]
