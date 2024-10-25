import pika
from .config import RABBIT_MQ_HOST, RABBIT_MQ_PASS, RABBIT_MQ_LOGIN


def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBIT_MQ_LOGIN, RABBIT_MQ_PASS)
    return pika.BlockingConnection(pika.ConnectionParameters(RABBIT_MQ_HOST, 5672, credentials=credentials))
