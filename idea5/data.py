"""Reuse the exact idea4 benchmark parsers and scorers."""
import hashlib
import json
from pathlib import Path


def load(config, root):
    from eval.finance.data_processor import DataProcessor
    processor = DataProcessor(config['task'])
    paths = config.get('data') or json.loads((root/'eval/finance/data/sample_config.json').read_text())[config['task']]
    splits, hashes = {}, {}
    for split in ('train', 'val', 'test'):
        path = Path(paths[split + '_data'])
        if not path.is_absolute():
            path = root/path
        raw = path.read_bytes()
        hashes[split] = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
        splits[split] = processor.process_task_data([json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()])
        if not splits[split]:
            raise ValueError('Empty split: ' + split)
        hashes[split]['samples'] = len(splits[split])
    return processor, splits, hashes
