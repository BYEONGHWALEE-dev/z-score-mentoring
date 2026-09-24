"""축소된 수업 로그에 대응하는 CERT r4.2 정답지를 생성한다."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 250_000
LOG_FILES = {
    "logon": "logon.csv",
    "device": "device.csv",
    "file": "file.csv",
    "http": "http.csv",
    "email": "email.csv",
}


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "Data" / "archive" / "answers" / "insiders.csv").is_file():
            return candidate
    raise FileNotFoundError("프로젝트의 원본 answers 디렉터리를 찾지 못했습니다.")


def load_r42_incidents(
    answers_dir: Path,
) -> tuple[list[dict[str, str]], list[str], dict[str, set[str]]]:
    with (answers_dir / "insiders.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        incidents = [row for row in reader if row["dataset"] == "4.2"]

    answer_ids: dict[str, set[str]] = defaultdict(set)
    for incident in incidents:
        scenario_dir = answers_dir / f"r4.2-{incident['scenario']}"
        with (scenario_dir / incident["details"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.reader(handle):
                if len(row) >= 2:
                    answer_ids[row[0]].add(row[1])
    return incidents, fieldnames, answer_ids


def find_retained_ids(
    class_data_dir: Path, answer_ids: dict[str, set[str]]
) -> tuple[dict[str, set[str]], set[str]]:
    retained: dict[str, set[str]] = defaultdict(set)
    all_users: set[str] = set()
    for log_type, filename in LOG_FILES.items():
        targets = answer_ids[log_type]
        for chunk in pd.read_csv(
            class_data_dir / filename,
            usecols=["id", "user"],
            dtype={"id": "string", "user": "string"},
            chunksize=CHUNK_SIZE,
        ):
            retained[log_type].update(
                chunk.loc[chunk["id"].isin(targets), "id"].dropna().tolist()
            )
            all_users.update(chunk["user"].dropna().tolist())
    return retained, all_users


def write_class_answers(
    answers_dir: Path,
    output_dir: Path,
    incidents: list[dict[str, str]],
    fieldnames: list[str],
    retained: dict[str, set[str]],
    all_users: set[str],
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("license.txt", "readme.txt", "scenarios.txt"):
        shutil.copy2(answers_dir / filename, output_dir / filename)

    with (output_dir / "insiders.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(incidents)

    insider_users = {incident["user"] for incident in incidents}
    with (output_dir / "insider_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["user", "is_insider"])
        writer.writeheader()
        writer.writerows(
            {
                "user": user,
                "is_insider": int(user in insider_users),
            }
            for user in sorted(all_users)
        )

    written_counts: dict[str, int] = defaultdict(int)
    for incident in incidents:
        scenario_name = f"r4.2-{incident['scenario']}"
        source = answers_dir / scenario_name / incident["details"]
        destination_dir = output_dir / scenario_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / incident["details"]

        with source.open(newline="", encoding="utf-8") as source_handle:
            rows = [
                row
                for row in csv.reader(source_handle)
                if len(row) >= 2 and row[1] in retained[row[0]]
            ]
        with destination.open("w", newline="", encoding="utf-8") as output_handle:
            csv.writer(output_handle).writerows(rows)
        for row in rows:
            written_counts[row[0]] += 1
    return written_counts


def main() -> None:
    root = find_project_root(Path(__file__).parent)
    answers_dir = root / "Data" / "archive" / "answers"
    class_data_dir = root / "Data" / "Data for Class"
    output_dir = root / "Data" / "Answers For Class"

    incidents, fieldnames, answer_ids = load_r42_incidents(answers_dir)
    retained, all_users = find_retained_ids(class_data_dir, answer_ids)
    written_counts = write_class_answers(
        answers_dir,
        output_dir,
        incidents,
        fieldnames,
        retained,
        all_users,
    )

    print(f"생성 위치: {output_dir}")
    print(f"r4.2 사건 수: {len(incidents)}")
    print(
        f"학생용 라벨: {len(all_users):,}명 중 "
        f"{len({incident['user'] for incident in incidents})}명 인사이더"
    )
    for log_type in LOG_FILES:
        total = len(answer_ids[log_type])
        written = written_counts[log_type]
        print(f"{log_type}: {written:,} / {total:,}개 정답 이벤트 보존")


if __name__ == "__main__":
    main()
