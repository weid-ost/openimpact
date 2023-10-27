import shutil
from hashlib import md5
from os import environ, makedirs
from pathlib import Path

import requests
from tqdm import tqdm


def get_data_home_folder(data_home_folder: str | Path = None) -> Path:
    if data_home_folder is None:
        data_home_folder = environ.get(
            "WEID_DATA_PATH", (Path.home() / ".weid_data")
        )

    data_home_folder = Path(data_home_folder).absolute()
    makedirs(data_home_folder, exist_ok=True)

    return data_home_folder


def download_from_url(url: str, download_path: str | Path):
    r = requests.get(url, stream=True, headers={"Accept-Encoding": None})
    r.raise_for_status()

    file_size = int(r.headers.get("content-length", 0))

    download_path = Path(download_path).absolute()

    desc = f"Downloading {url}"

    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with download_path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)


def checksum_ok(filename: str | Path, remote_checksum: str):
    filename = Path(filename)
    with open(filename, "rb") as myzip:
        checksum = md5(myzip.read()).hexdigest()

        if not checksum == remote_checksum:
            print("=================================================")
            print(f"Checksum wrong for {filename.name}")
            print(f"{checksum} != {remote_checksum}")
            print("=================================================")


def substring_match(include: list[str], string: str) -> str | None:
    for inc in include:
        if inc in string:
            return string
    return None


def zenodo_download(
    id: str | int,
    download_folder: str | Path,
    force_download: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
):
    url_zenodo = r"https://zenodo.org/api/records/"

    r = requests.get(f"{url_zenodo}{id}")

    r.raise_for_status()

    metadata = r.json()

    for file in metadata["files"]:
        file_key = file["key"]

        if include:
            file_key = substring_match(include, file_key)
            if file_key is None:
                continue

        if exclude:
            file_key_exclude = substring_match(exclude, file_key)
            if file_key_exclude is not None:
                continue

        filename = Path(download_folder).expanduser().absolute() / file_key

        if not filename.exists() or force_download:
            download_from_url(file["links"]["self"], filename)

        remote_checksum = file["checksum"].split(":")[-1]
        checksum_ok(filename, remote_checksum)
