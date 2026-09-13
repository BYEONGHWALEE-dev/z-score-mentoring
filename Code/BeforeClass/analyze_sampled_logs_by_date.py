from collections import Counter
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 500_000
SAMPLED_FILES = {
    "email": "email_sampled.csv",
    "http": "http_sampled.csv",
}


def find_project_root(start: Path) -> Path:
    """Data/Class 디렉터리가 있는 프로젝트 루트를 찾습니다."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "Data" / "Class").is_dir():
            return candidate
    raise FileNotFoundError("Data/Class 디렉터리를 찾을 수 없습니다.")


def count_logs_by_date(csv_path: Path) -> Counter:
    """CSV 전체를 메모리에 올리지 않고 날짜별 로그 수를 계산합니다."""
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    counts: Counter = Counter()
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=["date"],
            chunksize=CHUNK_SIZE,
            dtype={"date": "string"},
        ),
        start=1,
    ):
        dates = chunk["date"].str.slice(0, 10)
        valid_dates = dates.str.fullmatch(r"\d{2}/\d{2}/\d{4}", na=False)
        dates = dates.where(valid_dates, "INVALID_DATE")

        counts.update(
            {
                str(date): int(count)
                for date, count in dates.value_counts(dropna=False).items()
            }
        )
        total_rows += len(chunk)
        print(
            f"{csv_path.name}: {chunk_number}번째 청크, "
            f"누적 {total_rows:,}행 처리"
        )

    return counts


def build_daily_counts(counts_by_log: dict[str, Counter]) -> pd.DataFrame:
    """로그 종류별 Counter를 날짜 기준의 하나의 표로 합칩니다."""
    all_dates = sorted(
        set().union(*(counts.keys() for counts in counts_by_log.values())),
        key=lambda value: (
            value == "INVALID_DATE",
            pd.to_datetime(value, format="%m/%d/%Y", errors="coerce"),
        ),
    )

    result = pd.DataFrame({"date": all_dates})
    for log_name, counts in counts_by_log.items():
        result[f"{log_name}_count"] = [
            counts.get(date, 0) for date in all_dates
        ]

    count_columns = [f"{name}_count" for name in counts_by_log]
    result["total_count"] = result[count_columns].sum(axis=1)
    return result


def build_summary(daily_counts: pd.DataFrame) -> pd.DataFrame:
    """파일별 전체 및 일별 요약 통계를 만듭니다."""
    rows = []
    for log_name, filename in SAMPLED_FILES.items():
        counts = daily_counts[f"{log_name}_count"]
        rows.append(
            {
                "log_type": log_name,
                "source_file": filename,
                "total_logs": int(counts.sum()),
                "date_count": int((counts > 0).sum()),
                "min_logs_per_date": int(counts[counts > 0].min()),
                "mean_logs_per_date": float(counts[counts > 0].mean()),
                "max_logs_per_date": int(counts.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    project_root = find_project_root(Path(__file__).parent)
    data_dir = project_root / "Data" / "Class"

    counts_by_log = {
        log_name: count_logs_by_date(data_dir / filename)
        for log_name, filename in SAMPLED_FILES.items()
    }
    daily_counts = build_daily_counts(counts_by_log)
    summary = build_summary(daily_counts)

    daily_output = data_dir / "sampled_log_counts_by_date.csv"
    summary_output = data_dir / "sampled_log_counts_summary.csv"
    daily_counts.to_csv(daily_output, index=False)
    summary.to_csv(summary_output, index=False)

    print("\n파일별 요약")
    print(summary.to_string(index=False))
    print(f"\n날짜별 결과: {daily_output}")
    print(f"요약 결과: {summary_output}")


if __name__ == "__main__":
    main()
