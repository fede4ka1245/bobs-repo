from typing import Type
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
from starlette.responses import JSONResponse
from fastapi import Depends, HTTPException, status
from .database import connect_to_mongo
from .orm import Orm
from bson import ObjectId


def create_crud_routes(model: Type[Orm], collection_name: str, app):
    @app.post(f"/{collection_name}/", response_model=model)
    async def create_item(item: model, db: AsyncIOMotorDatabase = Depends(connect_to_mongo)):
        try:
            result = await db[collection_name].insert_one(item.dict())
            inserted_id = str(result.inserted_id)  # Convert ObjectId to string
            response_content = {**item.dict(), "id": inserted_id}

            return JSONResponse(content=response_content, status_code=status.HTTP_201_CREATED)
        except DuplicateKeyError:
            raise HTTPException(status_code=400, detail=f"{collection_name.capitalize()} with this ID already exists")

    @app.get(f"/{collection_name}/{{item_id}}/", response_model=model)
    async def read_item(item_id: str, db: AsyncIOMotorDatabase = Depends(connect_to_mongo)):
        item = await db[collection_name].find_one({"_id": ObjectId(item_id)})
        if item:
            return item
        raise HTTPException(status_code=404, detail=f"{collection_name.capitalize()} not found")

    @app.put(f"/{collection_name}/{{item_id}}/", response_model=model)
    async def update_item(item_id: str, updated_item: model, db: AsyncIOMotorDatabase = Depends(connect_to_mongo)):
        result = await db[collection_name].update_one({"_id": ObjectId(item_id)}, {"$set": updated_item.to_dict()})
        if result.modified_count == 1:
            return updated_item
        raise HTTPException(status_code=404, detail=f"{collection_name.capitalize()} not found")

    @app.delete(f"/{collection_name}/{{item_id}}/")
    async def delete_item(item_id: str, db: AsyncIOMotorDatabase = Depends(connect_to_mongo)):
        result = await db[collection_name].delete_one({"_id": ObjectId(item_id)})
        if result.deleted_count == 1:
            return JSONResponse(content={"message": f"{collection_name.capitalize()} deleted successfully"})
        raise HTTPException(status_code=404, detail=f"{collection_name.capitalize()} not found")