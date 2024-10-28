# Документация

## Запуск решения

1) Нужно скопировать .env.example
2) Убедитесь что у вас установлен docker. Запустите в корне проекта следующую команду: 
```
docker compose up -d
```
3) Осталось запустить ./ml-worker. Для работы на gpu его предлагается запускать отдельно.
Например, в jupiternotebook (см ./full_launch.ipynb)
4) Опционально*. если вы хотите запустить worker вместе с остальными сервисами, надо в файле docker-compose.yaml 
раскомментировать блок ml-worker

## Сервисы

Решение состоит из следующих сервисов:
- /gradio (чат бот)
- /ml-worker (мл воркер который отвечает на вопросы в чат-боте)
- rabbit-mq (см. docker-compose. брокер по которому общаются ml-worker и gradio)
- qdrant (см. docker-compose. Векторная база данных для RAG)
- nginx (см. docker-compose. сервис для проксирования решения)

## Pipline работы модели

Расположен в full_launch.ipyn


# ТРебуется дозагрузка модели

https://s3.timeweb.cloud/27d6b1d6-241f2159-8e91-4925-be63-7cc5c17ad8ac/DlYhGCdmbnpzaucWgPqkBZKXMiNQIUSAOLVowrtF.zip
