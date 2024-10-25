from typing import List, Optional, Dict, Any
from fastapi import UploadFile
from pydantic import BaseModel


class Orm(BaseModel):
    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


class PrinterSubscription(Orm):
    user_id: str
    printer_uid: str


class User(Orm):
    telegram_chat_id: int
    printers: Optional[List[int]] = []


class Printer(Orm):
    uid: str
    name: str
    description: Optional[str] = None


class Warn(Orm):
    rate: float
    reason: int


class Layer(Orm):
    order: int
    project_id: str
    printer_uid: str
    warns: Optional[List[Warn]] = []
    before_melting_image: Optional[UploadFile]
    after_melting_image: Optional[UploadFile]
    svg_image: Optional[UploadFile]


class Project(Orm):
    printer_uid: str
    name: str
    layers_len: Optional[int] = None
