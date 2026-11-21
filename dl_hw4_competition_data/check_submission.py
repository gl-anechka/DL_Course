"""Проверка CSV: python check_submission.py submission.csv.

Скрипт читает sample_submission.csv рядом с собой. Оценку качества не считает.
"""
import csv
import io
import math
from pathlib import Path
import sys


def check(path):
    root = Path(__file__).resolve().parent
    with (root / 'sample_submission.csv').open(encoding='utf-8', newline='') as stream:
        expected = {row['id'] for row in csv.DictReader(stream)}
    raw = Path(path).read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Файл больше 8 MiB')
    rows = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
    if next(rows, None) != ['id', 'prediction_string']:
        raise ValueError('Нужны две колонки: id,prediction_string')
    seen = set()
    for row in rows:
        if len(row) != 2:
            raise ValueError('В каждой строке должны быть два поля')
        key, value = row
        if key not in expected or key in seen:
            raise ValueError('Неизвестный или повторный ID: ' + key)
        seen.add(key)
        tokens = value.split()
        if len(tokens) % 6 or len(tokens) > 600:
            raise ValueError(key + ': нужны шестёрки значений, не более 100 рамок')
        for i in range(0, len(tokens), 6):
            if tokens[i] not in ('1', '2', '3', '4', '5', '6', '7'):
                raise ValueError(key + ': класс должен быть целым от 1 до 7')
            score, x1, y1, x2, y2 = map(float, tokens[i+1:i+6])
            if not all(math.isfinite(v) for v in (score, x1, y1, x2, y2)):
                raise ValueError(key + ': NaN и inf запрещены')
            if not (0 <= score <= 1 and 0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512):
                raise ValueError(key + ': неверный score или координаты xyxy')
    if seen != expected:
        raise ValueError('Не хватает test ID: ' + str(len(expected - seen)))
    return len(seen)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Использование: python check_submission.py submission.csv')
    try:
        print('OK:', check(sys.argv[1]), 'изображений')
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        raise SystemExit(str(error))
