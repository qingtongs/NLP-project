"""Download byte-verified official evaluation files without changing their contents."""
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

FILES = [
    ('qa_evaluate.py',
     'https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8/qa_evaluate.py',
     'a518bc37d94d2e84cc9012b41a758567e308dceea92aba14cda44a1364626ae7'),
    ('MultiHopRAG.json',
     'https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82/MultiHopRAG.json',
     '03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff'),
]

def main():
    for name, url, expected in FILES:
        path = Path(name)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                print(f'Already verified: {name}')
                continue
            raise RuntimeError(f'{name} exists but differs. Move it aside before downloading; it has not been overwritten.')
        print(f'Downloading {name} ...', flush=True)
        with urlopen(Request(url, headers={'User-Agent': 'NLP-course-eval'}), timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch: {name}; file was not saved.')
        path.write_bytes(data)
        print(f'Saved and verified: {name}')

if __name__ == '__main__':
    main()
