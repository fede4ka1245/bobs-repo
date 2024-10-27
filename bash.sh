#!/bin/bash

cp .env.example .env

git stash
git pull origin main
docker compose up -d --build
