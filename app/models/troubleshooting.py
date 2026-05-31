from enum import Enum

from pydantic import BaseModel


class GoodToGoStatus(Enum):
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"


class GoodToGoInfo(BaseModel):
    gtg: GoodToGoStatus
