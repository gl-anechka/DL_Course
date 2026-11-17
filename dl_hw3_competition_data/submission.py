"""RLE, CSV validation and local validation-set mIoU. Python + NumPy + Pillow."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SIDE = 384
LIMIT = 128 * 1024**2


def encode_rle(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected a nonempty 2D mask containing 0 and 1.')
    flat = np.pad(mask.astype(np.uint8).ravel(order='C'), (1, 1))
    edges = np.flatnonzero(flat[1:] != flat[:-1])
    return ' '.join(str(int(v)) for pair in zip(edges[::2] + 1, edges[1::2] - edges[::2]) for v in pair)


def decode_rle(text, height=SIDE, width=SIDE):
    if not isinstance(text, str) or type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError('Expected RLE text and positive integer dimensions.')
    fields = text.split()
    if len(fields) % 2 or any(not v.isascii() or not v.isdecimal() for v in fields):
        raise ValueError('RLE requires decimal start/length pairs.')
    flat = np.zeros(height * width, dtype=np.uint8)
    previous_end = 0
    for start, length in zip(fields[::2], fields[1::2]):
        start, length = int(start) - 1, int(length)
        end = start + length
        if start < previous_end or length < 1 or end > flat.size:
            raise ValueError('Invalid, overlapping or out-of-bounds RLE.')
        flat[start:end] = 1
        previous_end = end
    return flat.reshape(height, width)


def image_ids(split, data_dir=ROOT):
    if split not in ('train', 'val', 'test'):
        raise ValueError('Choose train, val or test.')
    ids = {p.stem for p in (Path(data_dir) / split / 'images').glob('*.jpg')}
    if not ids:
        raise ValueError('Image directory is empty or missing.')
    return ids


def read_predictions(path, split='test', data_dir=ROOT):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('CSV exceeds 128 MiB.')
    expected = image_ids(split, data_dir)
    csv.field_size_limit(LIMIT)
    predictions = {}
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ['id', 'rle']:
            raise ValueError('Expected exactly id,rle.')
        for row in reader:
            if len(row) != 2 or row[0] not in expected or row[0] in predictions:
                raise ValueError('Invalid row, unknown ID or duplicate ID.')
            predictions[row[0]] = decode_rle(row[1])
    if set(predictions) != expected:
        raise ValueError('Missing image IDs.')
    return predictions


def score_val(path, data_dir=ROOT):
    from PIL import Image
    predictions = read_predictions(path, 'val', data_dir)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for identifier, prediction in predictions.items():
        with Image.open(Path(data_dir) / 'val' / 'masks' / (identifier + '.png')) as im:
            target = np.array(im)
        if target.shape != (SIDE, SIDE) or not np.isin(target, [0, 1, 255]).all():
            raise ValueError('Expected a 384x384 target with values 0/1/255.')
        valid = target != 255
        matrix += np.bincount((2 * target[valid].astype(np.int64) + prediction[valid]).ravel(), minlength=4).reshape(2, 2)
    intersection = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    if not (union > 0).any():
        raise ValueError('No evaluated pixels.')
    iou = np.divide(intersection, union, out=np.full(2, np.nan), where=union > 0)
    return {'miou': float(iou[union > 0].mean()),
            'iou_background': float(iou[0]) if union[0] else None,
            'iou_fragment': float(iou[1]) if union[1] else None, 'confusion': matrix.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'score'])
    parser.add_argument('csv', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        if args.action == 'score':
            result = score_val(args.csv, args.data_dir)
        else:
            predictions = read_predictions(args.csv, 'test', args.data_dir)
            result = {'valid': True, 'images': len(predictions)}
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
