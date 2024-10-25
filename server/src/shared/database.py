from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from .config import *


async def connect_to_mongo():
    client = AsyncIOMotorClient(DATABASE_URL, username=MONGO_INITDB_ROOT_USERNAME, password=MONGO_INITDB_ROOT_PASSWORD)
    db = client[DATABASE_NAME]
    return db


async def close_mongo_connection(db: AsyncIOMotorDatabase):
    db.client.close()
