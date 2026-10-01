import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://zenodo.org/records/5907847/files/"
    "apachejit_dataset_replication.zip?download=1"
)


def fetch(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="apachejit-") as directory:
        archive = Path(directory) / "apachejit.zip"
        urllib.request.urlretrieve(DATASET_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            matches = [name for name in bundle.namelist() if name.endswith("apachejit_total.csv")]
            if len(matches) != 1:
                raise RuntimeError("The ApacheJIT archive did not contain one apachejit_total.csv")
            with bundle.open(matches[0]) as source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ApacheJIT v2 from its Zenodo record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.output)


if __name__ == "__main__":
    main()
