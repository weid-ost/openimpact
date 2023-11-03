from enum import Enum
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, conlist

from openimpact.utils import load_pickle
import numpy as np

app = FastAPI()

fake_items_db = [
    {"item_name": "Foo"},
    {"item_name": "Bar"},
    {"item_name": "Baz"},
]


class Model(str, Enum):
    gat = "gat"
    gcn = "gcn"
    mpnn = "mpnn"


class Item(BaseModel):
    name: str
    name_id: int
    address: str | None = None


class Input(BaseModel):
    wind_speed: float
    wind_direction: float
    yaw_misalignment: conlist(float, min_length=6, max_length=6)


class Output(BaseModel):
    adj_matrix: list[float]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/{name}")
async def say_hello(name: str):
    return {"Hello": name}


@app.get("/models/{model_name}")
async def get_model(model_name: Model):
    model_info = {
        "gat": "Graph Attention",
        "gcn": "Graph Convolution Network",
        "mpnn": "Message-passing Neural Network",
    }

    return {"Model": model_name, "message": model_info[model_name]}


@app.get("/db/")
async def get_db_entries(entry: int):
    return fake_items_db[entry]


@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]


@app.post("/test/")
async def create_item(item: Item):
    return item


@app.post("/input/", response_model=Output)
async def create_input(inputs: Input) -> Any:
    try:
        model = load_pickle("graph_models/graph_model.pkl")
        data = np.array(
            [
                inputs.wind_speed,
                inputs.wind_direction,
            ]
            + inputs.yaw_misalignment
        ).reshape(1, -1)

        pred_adj_mats = {"adj_matrix": np.ravel(model.predict(data)).tolist()}
    except Exception as inst:
        with open("std.err", "w") as f:
            print(inst, file=f)

    return pred_adj_mats
