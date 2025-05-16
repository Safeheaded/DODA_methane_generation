import requests
from concurrent.futures import ThreadPoolExecutor
import shutil
import os
from pathlib import Path
from tqdm import tqdm

urls = [
    "https://zenodo.org/records/7863343/files/STARCOP_test.zip?download=1",
       "https://zenodo.org/records/7863343/files/STARCOP_train_easy.zip?download=1",
       "https://zenodo.org/records/7863343/files/STARCOP_train_remaining_part1.zip?download=1",
       "https://zenodo.org/records/7863343/files/STARCOP_train_remaining_part2.zip?download=1",
       "https://zenodo.org/records/7863343/files/STARCOP_train_remaining_part3.zip?download=1",
       "https://zenodo.org/records/7863343/files/STARCOP_train_remaining_part4.zip?download=1",
       "https://zenodo.org/records/7863343/files/STARCOP_train_remaining_part5.zip?download=1"
]


def download_file(url):
    filename = url.split("/")[-1].split("?")[0]
    response = requests.get(url, stream=True)
    datasets_dir = Path(os.getcwd()) / "datasets" / "methane" / "data"
    datasets_dir.is_dir() or datasets_dir.mkdir(parents=True, exist_ok=True)
    filename = datasets_dir / filename

    # if os.listdir(datasets_dir):
    #     raise Exception(
    #         "Folder is not empty. Please remove files from the folder before downloading new ones."
    #     )

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(filename, "wb") as file, tqdm(total=total_size, unit="B", unit_scale=True, desc=filename.name) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

    shutil.unpack_archive(filename, datasets_dir)
    unpacked_path = filename.with_suffix("")
    for file in unpacked_path.iterdir():
        if file.is_dir():
            shutil.move(str(file), str(datasets_dir))
    os.remove(filename)
    shutil.rmtree(unpacked_path, ignore_errors=True)
    print(f"Pobrano: {filename}")


def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        _ = list(executor.map(download_file, urls))


if __name__ == "__main__":
    main()
