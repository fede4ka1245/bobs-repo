import aio_pika
import gradio as gr
import asyncio
import uuid
from aio_pika import IncomingMessage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RABBIT_MQ_HOST = 'rabbitmq'
RABBIT_MQ_PORT = '5672'
RABBITMQ_DEFAULT_USER = 'rmuser'
RABBITMQ_DEFAULT_PASS = 'rmpass'

queue = asyncio.Queue()

class RabbitMQClient:
    def __init__(self, user, password, host, port, queue_name):
        self._url = f"amqp://{user}:{password}@{host}:{port}/%2F"
        self._queue_name = queue_name
        self._connection = None
        self._channel = None
        self._queue = None

    async def connect(self):
        """Connect to RabbitMQ and configure a channel and queue."""
        try:
            self._connection = await aio_pika.connect_robust(self._url)
            self._channel = await self._connection.channel()

            # Ensure the queue exists
            self._queue = await self._channel.declare_queue(self._queue_name, durable=True)
            logger.info(f"Connected and declared queue '{self._queue_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def publish_message(self, message: str):
        """Publish a message to the declared queue."""
        if not self._channel or self._channel.is_closed:
            await self.connect()

        await self._channel.default_exchange.publish(
            aio_pika.Message(body=message.encode()),
            routing_key=self._queue_name
        )
        logger.info(f"Published message: {message}")

    async def close(self):
        if self._channel:
            await self._channel.close()
        if self._connection:
            await self._connection.close()

    async def consume_messages(self, process_callable):
        """Consume messages from the queue and process them."""
        if not self._channel or self._channel.is_closed:
            await self.connect()

        async def on_message(message: aio_pika.IncomingMessage):
            logger.info(f"Received message: {message.body.decode()}")
            async with message.process(ignore_processed=True):
                try:
                    process_callable(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Optionally: Reject the message if processing fails
                    # await message.reject()
                    return
                # Acknowledgement is automatic with `message.process()` context manager

        await self._queue.consume(on_message, no_ack=False)
        logger.info(f"Started consuming messages from '{self._queue_name}'")


async def listen_for_answer(msg_id, text, callback):
    user = RABBITMQ_DEFAULT_USER
    password = RABBITMQ_DEFAULT_PASS
    host = RABBIT_MQ_HOST
    port = RABBIT_MQ_PORT
    question_queue = 'ml_questions'
    answer_queue = 'ml_answers'

    question_client = RabbitMQClient(user, password, host, port, question_queue)
    answer_client = RabbitMQClient(user, password, host, port, answer_queue)

    await question_client.publish_message(json.dumps({"msg_id": msg_id, "text": text}))

    def process(message: IncomingMessage):
        try:
            message_body = json.loads(message.body.decode())

            callback({'text': message_body["text"], 'is_end': message_body['step'] == message_body['max_steps']})

            if message_body is None or message_body['msg_id'] != msg_id:
                return

            if message_body.get('step') == message_body.get('max_steps'):
                callback({'text': message_body['text'], 'is_end': message_body['step'] == message_body['max_steps']})
                asyncio.create_task(question_client.close())
                asyncio.create_task(answer_client.close())

        except Exception as error:
            print(error)
            callback({'text': 'К сожалению запрос обработался с ошибкой.', 'is_end': True})

    await answer_client.consume_messages(process)


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}, "id": uuid.uuid4() })
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"], "id": uuid.uuid4() })
    return history, gr.MultimodalTextbox(value=None, interactive=False)


async def bot(history: list):
    assistant_entry = {"role": "assistant", "content": "Обрабатываю ответ", "id": uuid.uuid4()}
    history.append(assistant_entry)

    queue = asyncio.Queue()

    def callback(obj):
        asyncio.create_task(queue.put(obj))

    history[-1]["content"] = "Обрабатываю ответ..."
    yield history

    await listen_for_answer(str(assistant_entry["id"]), assistant_entry["content"], callback)

    while True:
        message = await queue.get()

        history[-1]["content"] = message['text']
        yield history

        if message['is_end']:
            break


def save_text(input_text):
    global queue
    queue.put(input_text)
    print(input_text)

with gr.Blocks() as demo:
    # Creating a tab interface
    with gr.Tabs():
        # First tab: Main Chat interface
        with gr.Tab("Чат бот"):
            chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Enter message or upload file...",
                show_label=False,
            )

            chat_msg = chat_input.submit(
                add_message, [chatbot, chat_input], [chatbot, chat_input]
            )

            bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        # Second tab: Text form with Save button
        with gr.Tab("Записать данные"):
            text_input = gr.Textbox(
                lines=5,
                placeholder="Ввести текси...",
                label="Форма ввода"
            )
            save_button = gr.Button("Сохранить")

            # Connect the save button with the save_text function
            save_button.click(save_text, inputs=text_input)

    demo.queue(default_concurrency_limit=5)
    demo.launch(share=True)