FROM python:3.10-slim

WORKDIR /usr/src/app
COPY ./gradio .
RUN pip install -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "main.py"]