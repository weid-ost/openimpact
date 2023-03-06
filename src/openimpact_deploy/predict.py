import argparse
import time
import numpy as np
import requests
import json


def predict():
    parser = argparse.ArgumentParser(description="My program")
    parser.add_argument("-f", dest="json", help="json file")
    args = parser.parse_args()

    if args.json:
        with open(f"{args.json}", "r") as f:
            json_file = json.load(f)
    else:
        json_file = {
            "wind_speed": 10,
            "wind_direction": 180.0,
            "yaw_misalignment": [1, 2, 3, 4],
        }

    result = requests.post(
        "http://127.0.0.1:8000/input/",
        json=json_file,
    )
    print(result.text)


def stream_data():
    yield {
        "wind_speed": np.random.random(),
        "wind_direction": np.random.random(),
        "yaw_misalignment": np.random.random(6).tolist(),
    }


def predict_stream():
    for i in range(10):
        data = next(stream_data())
        result = requests.post(
            "http://127.0.0.1:8000/input/",
            json=data,
        )
        print(result.text)
        time.sleep(2)
