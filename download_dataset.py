"""Download the UCI Phishing Websites CSV mirror used in the project report."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/vaibhavbichave/Phishing-URL-Detection/master/phishing.csv"
DEST = Path(__file__).parent / "data" / "uci_phishing.csv"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset (11,054 rows; 30 features) ...")
    try:
        with urlopen(URL, timeout=30) as response:
            content = response.read()
        if len(content) < 100_000:
            raise RuntimeError("Downloaded file is unexpectedly small.")
        DEST.write_bytes(content)
        print(f"Saved dataset to: {DEST}")
        print("Source mirror: vaibhavbichave/Phishing-URL-Detection; original dataset: UCI ID 327.")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Download the UCI Phishing Websites dataset manually and save its CSV as:")
        print(DEST)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
