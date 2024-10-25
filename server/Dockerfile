FROM python:3.11
WORKDIR /app/src
COPY ./requirements.txt /app/src
COPY ./src .

RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "main.py" ]