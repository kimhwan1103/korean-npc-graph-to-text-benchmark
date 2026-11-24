# JSONL 입력을 읽고, 쓰다가 중단돼도 기존 산출물이 남도록 임시 파일을 사용한다.
"""Read JSONL and publish complete dataset files atomically."""

import json
import os
from tempfile import NamedTemporaryFile


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # 입력 파일과 줄 번호를 같이 남겨 잘못된 행을 찾기 쉽게 한다.
                    raise ValueError(f"{path}:{number}: {exc.msg}") from exc
    return rows


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # 임시 파일을 출력 경로와 같은 디렉터리에 만들고 저장이 끝난 뒤 한 번에 교체한다.
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = file.name
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
