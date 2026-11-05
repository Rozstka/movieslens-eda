import hashlib
import ssl
import certifi
import io
from pathlib import Path
import urllib.request
import zipfile
URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
# GroupLens publishes this MD5 alongside the stable archive. Integrity check, not a security signature.
EXPECTED_MD5 = '0e33842e24a9c977be4e0107933c0723'
def main():
    directory = Path(__file__).resolve().parents[1] / 'data'
    directory.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
        data = response.read(10_000_001)
    if len(data) > 10_000_000 or hashlib.md5(data).hexdigest() != EXPECTED_MD5:
        raise SystemExit('Archive checksum mismatch; inspect upstream before updating the pinned checksum')
    # Extract only known files, never arbitrary archive paths.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in ('u.data', 'u.item', 'README'):
            (directory / name).write_bytes(archive.read('ml-100k/' + name))
    (directory / 'SOURCE.txt').write_text(URL + '\nsha256=' + hashlib.sha256(data).hexdigest() + '\n')
    print('Downloaded verified MovieLens 100K into data/. Read data/README for usage terms.')
if __name__ == '__main__': main()
