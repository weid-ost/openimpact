from .download import download_from_url, zenodo_download
from zipfile import ZipFile
from pathlib import Path
import pandas as pd


def kelmarsh_raw_data(download_path: str | Path) -> None:
    download_path = Path(download_path)

    download_path.mkdir(exist_ok=True, parents=True)

    zenodo_download(
        8252025,
        download_path,
        force_download=False,
        include=["Kelmarsh_WT_static.csv", "Kelmarsh_SCADA_2022_4457.zip"],
    )

    raw_csv = download_path / "kelmarsh_raw.csv"

    if not raw_csv.exists():
        df = _scada_zip_to_dataframe(
            download_path / "Kelmarsh_SCADA_2022_4457.zip",
            filter_exp="Turbine_Data",
            skiprows=9,
        )

        df.to_csv(raw_csv, index=False)


def _scada_zip_to_dataframe(
    filename: str | Path, filter_exp: str = "", skiprows: int = 0
) -> pd.DataFrame:
    """Accepts a zip file that contains csv SCADA files."""

    with ZipFile(filename) as myzip:
        data_files = [f for f in myzip.namelist() if filter_exp in f]

        frames = []
        for f in data_files:
            wind_turbine = int(Path(f).stem.split("_")[-1])
            with myzip.open(f, "r") as wt:
                df_tmp = pd.read_csv(wt, skiprows=skiprows)
                df_tmp["Wind turbine ID"] = wind_turbine
                frames.append(df_tmp)

    return pd.concat(frames)
