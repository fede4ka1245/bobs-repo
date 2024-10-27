import asyncio
import logging
import tracemalloc
from aio_pika import IncomingMessage, connect_robust, Message, ExchangeType
import json
import logging
import time

from ml import get_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RABBIT_MQ_HOST = '213.171.5.51'
RABBIT_MQ_PORT = '5672'
RABBITMQ_DEFAULT_USER = 'rmuser'
RABBITMQ_DEFAULT_PASS = 'rmpass'

from rag import get_context
from saiga import get_answer


class RabbitMQClient:
    def __init__(self, user, password, host, port, queue_name):
        self._url = f"amqp://{user}:{password}@{host}:{port}/%2F"
        self._queue_name = queue_name
        self._connection = None
        self._channel = None
        self._queue = None

    async def connect(self):
        """Connect to RabbitMQ and configure a channel and queue."""
        self._connection = await connect_robust(self._url)
        self._channel = await self._connection.channel()

        # Ensure the queue exists
        self._queue = await self._channel.declare_queue(self._queue_name, durable=True)
        logger.info(f"Connected and declared queue '{self._queue_name}'")

    async def publish_message(self, message: str):
        """Publish a message to the declared queue."""
        if not self._channel or self._channel.is_closed:
            await self.connect()

        await self._channel.default_exchange.publish(
            Message(body=message.encode()),
            routing_key=self._queue_name
        )
        logger.info(f"Published message: {message}")

    async def consume_messages(self, process_callable):
        """Consume messages from the queue and process them."""
        if not self._channel or self._channel.is_closed:
            await self.connect()

        async def on_message(message: IncomingMessage):
            logger.info(f"Started consuming messages from '{self._queue_name}'")
            try:
                async with message.process():
                    await process_callable(message)
            except Exception as e:
                logger.error(f"Error processing message: {e}")

        await self._queue.consume(on_message)
        logger.info(f"Started consuming messages from '{self._queue_name}'")


async def worker(user, password, host, port, question_queue, answer_queue):
    question_client = RabbitMQClient(user, password, host, port, question_queue)
    answer_client = RabbitMQClient(user, password, host, port, answer_queue)

    await question_client.connect()
    await answer_client.connect()

    async def pipeline(msg):
        async def send_step(step_data):
            await answer_client.publish_message(json.dumps(step_data))
            logger.info(f"Sent step: {step_data}")

        # question = msg['text']
        # step1 = {'max_steps': 3, 'step': 1, 'msg': "Step 1", 'msg_id': msg['msg_id']}
        # await send_step(step1)
        #
        # context = get_context(question, 10)
        # step2 = {'max_steps': 3, 'step': 2, 'msg': "Step 2", 'msg_id': msg['msg_id']}
        # await send_step(step2)
        #
        # answer = get_answer(question, context)
        # step3 = {'max_steps': 3, 'step': 3, 'msg': answer, 'msg_id': msg['msg_id']}
        # await send_step(step3)

        step1 = {'max_steps': 1, 'step': 1, 'msg': get_result(msg, "FULL_UPLOAD"), 'msg_id': msg['msg_id']}
        await send_step(step1)

    async def process(message: IncomingMessage):
        logger.info("Received message:")
        print(f"Message Bodies: {message.body}")
        message_body = json.loads(message.body.decode())
        logger.info(f"Received message: {message_body}")

        await pipeline(message_body)

    await question_client.consume_messages(process)


async def main():
    user = RABBITMQ_DEFAULT_USER
    password = RABBITMQ_DEFAULT_PASS
    host = RABBIT_MQ_HOST
    port = RABBIT_MQ_PORT
    question_queue = 'ml_questions3'
    answer_queue = 'ml_answers3'

    try:
        await worker(user, password, host, port, question_queue, answer_queue)
    except Exception as e:
        logger.error(f"Worker terminated due to an error: {e}")

    while True:
        await asyncio.sleep(1)


if __name__== "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
