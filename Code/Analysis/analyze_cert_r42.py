"""CERT r4.2 멀티 로그 이상탐지와 정답지 사후 검증.

원본 content 열을 읽지 않고 5종 로그를 사용자-일자 feature로 집계한다.
평가일 이전 60일만 사용하는 개인별 rolling baseline으로 양의 z-score를
계산하고, answers는 점수 계산이 끝난 뒤 평가에만 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
ROLLING_DAYS = 60
MIN_BASELINE_DAYS = 14
ZERO_STD_Z = 6.0
Z_CAP = 10.0
CHUNK_SIZE = 250_000

GENERIC_WEIGHTS = {
    "after_hours_logon_count": 1.0,
    "logon_pc_count": 0.6,
    "logon_count": 0.3,
    "after_hours_device_count": 1.0,
    "device_connect_count": 0.4,
    "usb_file_count": 1.4,
    "file_count": 0.7,
    "unique_file_count": 0.5,
    "after_hours_file_count": 0.8,
    "http_domain_count": 0.5,
    "after_hours_http_count": 0.6,
    "external_email_count": 0.8,
    "external_attachment_email_count": 1.0,
    "external_email_size": 0.8,
    "after_hours_email_count": 0.6,
    "mass_recipient_email_count": 1.0,
}

# 위협 가설에 기반한 의미 feature이다. answers를 이용해 가중치를 최적화하지 않는다.
SEMANTIC_WEIGHTS = {
    "job_site_count": 1.0,
    "leak_site_count": 1.4,
    "keylogger_site_count": 1.4,
    "usb_executable_count": 1.4,
}

LOG_SPECS = {
    "logon": ("logon.csv", ["id", "date", "user", "pc", "activity"]),
    "device": ("device.csv", ["id", "date", "user", "pc", "activity"]),
    "file": ("file.csv", ["id", "date", "user", "pc", "filename"]),
    "http": ("http.csv", ["id", "date", "user", "pc", "url"]),
    "email": (
        "email.csv",
        [
            "id",
            "date",
            "user",
            "pc",
            "to",
            "cc",
            "bcc",
            "from",
            "size",
            "attachments",
        ],
    ),
}


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "Data" / "archive" / "answers" / "insiders.csv").is_file():
            return candidate
    raise FileNotFoundError("Data/archive/answers/insiders.csv를 찾을 수 없습니다.")


def find_data_dir(root: Path) -> Path:
    candidates = [root / "Data" / "Data for Class", root / "Data" / "Class"]
    for candidate in candidates:
        if all((candidate / spec[0]).is_file() for spec in LOG_SPECS.values()):
            return candidate
    expected = ", ".join(filename for filename, _ in LOG_SPECS.values())
    raise FileNotFoundError(f"5종 수업 로그를 한 폴더에서 찾지 못했습니다: {expected}")


def parse_time(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["event_time"] = pd.to_datetime(
        frame.pop("date"), format=DATE_FORMAT, errors="coerce"
    )
    invalid = int(frame["event_time"].isna().sum())
    if invalid:
        raise ValueError(f"파싱할 수 없는 date가 {invalid:,}개 있습니다.")
    frame["event_date"] = frame["event_time"].dt.normalize()
    frame["after_hours"] = (
        (frame["event_time"].dt.dayofweek >= 5)
        | (frame["event_time"].dt.hour < 7)
        | (frame["event_time"].dt.hour >= 19)
    )
    return frame


def grouped_feature(
    frame: pd.DataFrame, column: str, operation: str = "sum"
) -> pd.Series:
    grouped = frame.groupby(["user", "event_date"], observed=True)[column]
    if operation == "sum":
        return grouped.sum()
    if operation == "count":
        return grouped.count()
    if operation == "nunique":
        return grouped.nunique()
    raise ValueError(operation)


def load_answers(answers_dir: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    insiders = pd.read_csv(answers_dir / "insiders.csv", dtype={"dataset": "string"})
    insiders = insiders.loc[insiders["dataset"].eq("4.2")].copy()
    insiders["start"] = pd.to_datetime(insiders["start"], format=DATE_FORMAT)
    insiders["end"] = pd.to_datetime(insiders["end"], format=DATE_FORMAT)

    answer_ids: dict[str, set[str]] = defaultdict(set)
    event_counts: list[int] = []
    type_counts: list[str] = []
    for details in insiders["details"]:
        path = answers_dir / f"r4.2-{str(details).split('-')[1]}" / details
        incident_types: defaultdict[str, int] = defaultdict(int)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                log_type, event_id = row[0], row[1]
                answer_ids[log_type].add(event_id)
                incident_types[log_type] += 1
        event_counts.append(sum(incident_types.values()))
        type_counts.append(
            ";".join(f"{key}:{value}" for key, value in sorted(incident_types.items()))
        )
    insiders["answer_event_count"] = event_counts
    insiders["answer_event_types"] = type_counts
    return insiders.reset_index(drop=True), answer_ids


def read_chunks(
    path: Path,
    columns: list[str],
    target_answer_ids: set[str],
    covered_answer_ids: dict[str, set[str]],
    log_type: str,
):
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        covered_answer_ids[log_type].update(
            chunk.loc[chunk["id"].isin(target_answer_ids), "id"].tolist()
        )
        yield parse_time(chunk)


def concat_feature_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts)
    return combined.groupby(level=[0, 1], observed=True).sum()


def aggregate_logon(
    path: Path, answer_ids: set[str], covered: dict[str, set[str]]
) -> pd.DataFrame:
    parts = []
    for chunk in read_chunks(
        path, LOG_SPECS["logon"][1], answer_ids, covered, "logon"
    ):
        chunk = chunk.loc[chunk["activity"].eq("Logon")]
        part = pd.concat(
            {
                "logon_count": grouped_feature(chunk, "id", "count"),
                "after_hours_logon_count": grouped_feature(
                    chunk, "after_hours", "sum"
                ),
                "logon_pc_count": grouped_feature(chunk, "pc", "nunique"),
            },
            axis=1,
        )
        parts.append(part)
    return concat_feature_parts(parts)


def aggregate_device(
    path: Path, answer_ids: set[str], covered: dict[str, set[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = list(
        read_chunks(path, LOG_SPECS["device"][1], answer_ids, covered, "device")
    )
    events = pd.concat(chunks, ignore_index=True)
    connects = events.loc[events["activity"].eq("Connect")]
    features = pd.concat(
        {
            "device_connect_count": grouped_feature(connects, "id", "count"),
            "after_hours_device_count": grouped_feature(
                connects, "after_hours", "sum"
            ),
            "device_pc_count": grouped_feature(connects, "pc", "nunique"),
        },
        axis=1,
    )
    return features, events[["user", "pc", "event_time", "activity"]]


def file_in_usb_sessions(
    file_events: pd.DataFrame, device_events: pd.DataFrame
) -> np.ndarray:
    """동일 user/pc에서 Connect 다음 Disconnect 사이인 file 이벤트를 표시한다."""
    sessions = device_events.sort_values(["user", "pc", "event_time"]).copy()
    sessions["end_time"] = sessions.groupby(["user", "pc"], observed=True)[
        "event_time"
    ].shift(-1)
    sessions["next_activity"] = sessions.groupby(["user", "pc"], observed=True)[
        "activity"
    ].shift(-1)
    sessions = sessions.loc[
        sessions["activity"].eq("Connect")
        & sessions["next_activity"].eq("Disconnect")
        & sessions["end_time"].notna(),
        ["user", "pc", "event_time", "end_time"],
    ]

    result = np.zeros(len(file_events), dtype=bool)
    session_groups = {
        key: (
            group["event_time"].to_numpy(dtype="datetime64[ns]"),
            group["end_time"].to_numpy(dtype="datetime64[ns]"),
        )
        for key, group in sessions.groupby(["user", "pc"], observed=True)
    }
    for key, group in file_events.groupby(["user", "pc"], observed=True):
        bounds = session_groups.get(key)
        if bounds is None:
            continue
        starts, ends = bounds
        times = group["event_time"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(starts, times, side="right") - 1
        valid = positions >= 0
        safe_positions = np.maximum(positions, 0)
        valid &= times <= ends[safe_positions]
        result[group.index.to_numpy()] = valid
    return result


def aggregate_file(
    path: Path,
    answer_ids: set[str],
    covered: dict[str, set[str]],
    device_events: pd.DataFrame,
) -> pd.DataFrame:
    chunks = list(read_chunks(path, LOG_SPECS["file"][1], answer_ids, covered, "file"))
    events = pd.concat(chunks, ignore_index=True)
    events["is_executable"] = events["filename"].str.lower().str.endswith(".exe")
    events["during_usb"] = file_in_usb_sessions(events, device_events)
    events["usb_executable"] = events["during_usb"] & events["is_executable"]
    return pd.concat(
        {
            "file_count": grouped_feature(events, "id", "count"),
            "unique_file_count": grouped_feature(events, "filename", "nunique"),
            "after_hours_file_count": grouped_feature(events, "after_hours", "sum"),
            "usb_file_count": grouped_feature(events, "during_usb", "sum"),
            "usb_executable_count": grouped_feature(
                events, "usb_executable", "sum"
            ),
        },
        axis=1,
    )


def extract_domain(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return urlparse(text).netloc.lower().split(":")[0]


def aggregate_http(
    path: Path, answer_ids: set[str], covered: dict[str, set[str]]
) -> pd.DataFrame:
    parts = []
    job_tokens = (
        "career",
        "job",
        "monster.",
        "linkedin.",
        "indeed.",
        "simplyhired.",
        "craigslist.",
        "boeing.",
        "lockheedmartin.",
        "northropgrumman.",
    )
    leak_tokens = ("wikileaks.", "dropbox.", "mediafire.", "megaupload.")
    keylogger_tokens = ("keylog",)
    for chunk in read_chunks(path, LOG_SPECS["http"][1], answer_ids, covered, "http"):
        chunk["domain"] = chunk["url"].map(extract_domain)
        chunk["job_site"] = chunk["domain"].str.contains(
            "|".join(job_tokens), regex=True, na=False
        )
        chunk["leak_site"] = chunk["domain"].str.contains(
            "|".join(leak_tokens), regex=True, na=False
        )
        chunk["keylogger_site"] = chunk["url"].str.contains(
            "|".join(keylogger_tokens), case=False, regex=True, na=False
        )
        parts.append(
            pd.concat(
                {
                    "http_count": grouped_feature(chunk, "id", "count"),
                    "http_domain_count": grouped_feature(chunk, "domain", "nunique"),
                    "after_hours_http_count": grouped_feature(
                        chunk, "after_hours", "sum"
                    ),
                    "job_site_count": grouped_feature(chunk, "job_site", "sum"),
                    "leak_site_count": grouped_feature(chunk, "leak_site", "sum"),
                    "keylogger_site_count": grouped_feature(
                        chunk, "keylogger_site", "sum"
                    ),
                },
                axis=1,
            )
        )
    return concat_feature_parts(parts)


def count_recipients(value: object) -> int:
    if pd.isna(value) or not str(value).strip():
        return 0
    return len([item for item in str(value).split(";") if item.strip()])


def has_external_recipient(row: pd.Series) -> bool:
    recipients = ";".join(
        "" if pd.isna(row[column]) else str(row[column])
        for column in ("to", "cc", "bcc")
    )
    addresses = [item.strip().lower() for item in recipients.split(";") if item.strip()]
    return any(
        "@" in address and not address.endswith("@dtaa.com") for address in addresses
    )


def aggregate_email(
    path: Path, answer_ids: set[str], covered: dict[str, set[str]]
) -> pd.DataFrame:
    parts = []
    for chunk in read_chunks(path, LOG_SPECS["email"][1], answer_ids, covered, "email"):
        chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce").fillna(0)
        chunk["attachments"] = pd.to_numeric(
            chunk["attachments"], errors="coerce"
        ).fillna(0)
        chunk["external"] = chunk.apply(has_external_recipient, axis=1)
        chunk["external_attachment"] = chunk["external"] & chunk["attachments"].gt(0)
        chunk["external_size"] = chunk["size"].where(chunk["external"], 0)
        chunk["recipient_count"] = sum(
            chunk[column].map(count_recipients) for column in ("to", "cc", "bcc")
        )
        chunk["mass_recipient"] = chunk["recipient_count"].ge(20)
        parts.append(
            pd.concat(
                {
                    "email_count": grouped_feature(chunk, "id", "count"),
                    "external_email_count": grouped_feature(
                        chunk, "external", "sum"
                    ),
                    "external_attachment_email_count": grouped_feature(
                        chunk, "external_attachment", "sum"
                    ),
                    "external_email_size": grouped_feature(
                        chunk, "external_size", "sum"
                    ),
                    "after_hours_email_count": grouped_feature(
                        chunk, "after_hours", "sum"
                    ),
                    "mass_recipient_email_count": grouped_feature(
                        chunk, "mass_recipient", "sum"
                    ),
                },
                axis=1,
            )
        )
    return concat_feature_parts(parts)


def build_daily_grid(feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    observed = pd.concat(feature_frames, axis=1).fillna(0).reset_index()
    users = observed["user"].unique()
    all_dates = pd.date_range(
        observed["event_date"].min(), observed["event_date"].max(), freq="D"
    )
    grid = pd.MultiIndex.from_product(
        [users, all_dates], names=["user", "event_date"]
    ).to_frame(index=False)
    bounds = observed.groupby("user", observed=True)["event_date"].agg(["min", "max"])
    grid = grid.join(bounds, on="user")
    grid = grid.loc[
        grid["event_date"].between(grid["min"], grid["max"])
    ].drop(columns=["min", "max"])
    return (
        grid.merge(observed, on=["user", "event_date"], how="left")
        .fillna(0)
        .sort_values(["user", "event_date"])
        .reset_index(drop=True)
    )


def add_z_scores(daily: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    scored = daily.copy()
    for feature in feature_names:
        transformed = np.log1p(scored[feature].astype(float))
        shifted = transformed.groupby(scored["user"], observed=True).shift(1)
        baseline_group = shifted.groupby(scored["user"], observed=True)
        mean = baseline_group.transform(
            lambda values: values.rolling(
                ROLLING_DAYS, min_periods=MIN_BASELINE_DAYS
            ).mean()
        )
        std = baseline_group.transform(
            lambda values: values.rolling(
                ROLLING_DAYS, min_periods=MIN_BASELINE_DAYS
            ).std(ddof=0)
        )
        delta = transformed - mean
        z = np.where(
            mean.isna(),
            0.0,
            np.where(
                std.gt(1e-12),
                delta / std,
                np.where(delta.gt(0), ZERO_STD_Z, 0.0),
            ),
        )
        scored[f"z_{feature}"] = np.clip(z, 0, Z_CAP)
    return scored


def weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    denominator = sum(weights.values())
    score = sum(frame[f"z_{name}"] * weight for name, weight in weights.items())
    return score / denominator


def build_user_ranking(scored: pd.DataFrame) -> pd.DataFrame:
    peak_indices = scored.groupby("user", observed=True)["hybrid_score"].idxmax()
    ranking = scored.loc[
        peak_indices,
        ["user", "event_date", "generic_score", "hybrid_score"],
    ].rename(columns={"event_date": "peak_date"})
    generic = (
        scored.groupby("user", observed=True)["generic_score"]
        .max()
        .rename("generic_peak_score")
    )
    ranking = ranking.join(generic, on="user")
    ranking["hybrid_rank"] = ranking["hybrid_score"].rank(
        method="min", ascending=False
    ).astype(int)
    ranking["generic_rank"] = ranking["generic_peak_score"].rank(
        method="min", ascending=False
    ).astype(int)
    return ranking.sort_values(["hybrid_rank", "user"]).reset_index(drop=True)


def explain_peaks(
    scored: pd.DataFrame, ranking: pd.DataFrame, all_weights: dict[str, float]
) -> pd.Series:
    keys = ranking.set_index(["user", "peak_date"])
    peak_rows = scored.set_index(["user", "event_date"]).loc[keys.index]
    explanations = []
    for _, row in peak_rows.iterrows():
        contributions = sorted(
            (
                (
                    name,
                    float(row[f"z_{name}"]) * weight,
                    float(row[name]),
                    float(row[f"z_{name}"]),
                )
                for name, weight in all_weights.items()
                if row[f"z_{name}"] > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        explanations.append(
            "; ".join(
                f"{name}={value:g} (z={z_value:.2f})"
                for name, _, value, z_value in contributions
            )
            or "양의 z-score 없음"
        )
    return pd.Series(explanations, index=ranking.index, name="top_reasons")


def explain_row(row: pd.Series, all_weights: dict[str, float]) -> str:
    contributions = sorted(
        (
            (
                name,
                float(row[f"z_{name}"]) * weight,
                float(row[name]),
                float(row[f"z_{name}"]),
            )
            for name, weight in all_weights.items()
            if row[f"z_{name}"] > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    return (
        "; ".join(
            f"{name}={value:g} (z={z_value:.2f})"
            for name, _, value, z_value in contributions
        )
        or "양의 z-score 없음"
    )


def evaluate(
    insiders: pd.DataFrame,
    ranking: pd.DataFrame,
    scored: pd.DataFrame,
    all_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluated = insiders.merge(ranking, on="user", how="left")
    incident_rows = []
    for incident in evaluated.itertuples():
        user_days = scored.loc[scored["user"].eq(incident.user)]
        in_period = user_days["event_date"].between(
            incident.start.normalize(), incident.end.normalize()
        )
        period = user_days.loc[in_period]
        if period.empty:
            period_max = 0.0
            period_peak_date = pd.NaT
            period_reasons = "해당 기간의 축소 로그 없음"
        else:
            period_peak_row = period.loc[period["hybrid_score"].idxmax()]
            period_max = float(period_peak_row["hybrid_score"])
            period_peak_date = period_peak_row["event_date"]
            period_reasons = explain_row(period_peak_row, all_weights)
        percentile = (
            float((user_days["hybrid_score"] <= period_max).mean())
            if not user_days.empty
            else np.nan
        )
        incident_rows.append(
            {
                "details": incident.details,
                "answer_period_peak_score": period_max,
                "answer_period_peak_date": period_peak_date,
                "answer_period_top_reasons": period_reasons,
                "answer_period_user_percentile": percentile,
                "overall_peak_in_answer_period": bool(
                    pd.notna(incident.peak_date)
                    and incident.start.normalize()
                    <= incident.peak_date
                    <= incident.end.normalize()
                ),
                "peak_days_from_incident_end": (
                    int((incident.peak_date - incident.end.normalize()).days)
                    if pd.notna(incident.peak_date)
                    else np.nan
                ),
            }
        )
    evaluated = evaluated.merge(pd.DataFrame(incident_rows), on="details", how="left")

    metrics = []
    insider_users = set(insiders["user"])
    for score_name, rank_column in (
        ("generic", "generic_rank"),
        ("hybrid", "hybrid_rank"),
    ):
        for k in (10, 20, 50, 100):
            top_users = set(
                ranking.nsmallest(min(k, len(ranking)), rank_column)["user"]
            )
            hits = len(top_users & insider_users)
            metrics.append(
                {
                    "score": score_name,
                    "k": k,
                    "hits": hits,
                    "precision_at_k": hits / min(k, len(ranking)),
                    "recall_at_k": hits / len(insider_users),
                }
            )
    metric_frame = pd.DataFrame(metrics)
    scenario = (
        evaluated.groupby("scenario", observed=True)
        .agg(
            insiders=("user", "nunique"),
            median_hybrid_rank=("hybrid_rank", "median"),
            top_10=("hybrid_rank", lambda values: int((values <= 10).sum())),
            top_50=("hybrid_rank", lambda values: int((values <= 50).sum())),
            peak_in_period=("overall_peak_in_answer_period", "sum"),
            median_period_percentile=("answer_period_user_percentile", "median"),
        )
        .reset_index()
    )
    return evaluated, metric_frame, scenario


def answer_coverage_frame(
    answer_ids: dict[str, set[str]], covered: dict[str, set[str]]
) -> pd.DataFrame:
    rows = []
    for log_type in sorted(answer_ids):
        total = len(answer_ids[log_type])
        present = len(answer_ids[log_type] & covered[log_type])
        rows.append(
            {
                "log_type": log_type,
                "answer_event_ids": total,
                "present_in_class_data": present,
                "coverage": present / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def scan_answer_coverage(
    data_dir: Path, answer_ids: dict[str, set[str]]
) -> dict[str, set[str]]:
    """점수 확정 후에만 정답 이벤트가 축소 로그에 남았는지 검사한다."""
    covered: dict[str, set[str]] = defaultdict(set)
    for log_type, (filename, _) in LOG_SPECS.items():
        targets = answer_ids[log_type]
        for chunk in pd.read_csv(
            data_dir / filename,
            usecols=["id"],
            chunksize=CHUNK_SIZE,
            dtype={"id": "string"},
        ):
            covered[log_type].update(
                chunk.loc[chunk["id"].isin(targets), "id"].dropna().tolist()
            )
    return covered


def build_blind_review(data_dir: Path, ranking: pd.DataFrame) -> pd.DataFrame:
    """answers 없이 상위 후보의 peak 이전 최신 LDAP 직무 맥락을 붙인다."""
    snapshots = sorted((data_dir / "LDAP").glob("????-??.csv"))
    if not snapshots:
        return ranking.nsmallest(50, "hybrid_rank").copy()
    ldap_parts = []
    for snapshot in snapshots:
        part = pd.read_csv(
            snapshot,
            usecols=[
                "user_id",
                "role",
                "functional_unit",
                "department",
                "team",
            ],
        ).rename(columns={"user_id": "user"})
        part["ldap_month"] = pd.Period(snapshot.stem, freq="M")
        ldap_parts.append(part)
    ldap = pd.concat(ldap_parts, ignore_index=True)
    review = ranking.nsmallest(50, "hybrid_rank").copy()
    review["peak_month"] = review["peak_date"].dt.to_period("M")
    context_rows = []
    for row in review.itertuples():
        history = ldap.loc[
            ldap["user"].eq(row.user) & ldap["ldap_month"].le(row.peak_month)
        ].sort_values("ldap_month")
        if history.empty:
            context_rows.append({})
        else:
            context_rows.append(
                history.iloc[-1][
                    [
                        "role",
                        "functional_unit",
                        "department",
                        "team",
                        "ldap_month",
                    ]
                ].to_dict()
            )
    return pd.concat(
        [review.reset_index(drop=True), pd.DataFrame(context_rows)], axis=1
    )


def analyze_ldap(
    data_dir: Path, insiders: pd.DataFrame
) -> pd.DataFrame:
    ldap_dir = data_dir / "LDAP"
    snapshots = sorted(ldap_dir.glob("????-??.csv"))
    if not snapshots:
        return pd.DataFrame(
            columns=[
                "details",
                "ldap_last_present_month",
                "ldap_first_absent_month",
                "absence_month_offset_from_incident_end",
            ]
        )
    membership = {
        pd.Period(path.stem, freq="M"): set(
            pd.read_csv(path, usecols=["user_id"])["user_id"]
        )
        for path in snapshots
    }
    rows = []
    for incident in insiders.itertuples():
        present_months = sorted(
            month for month, users in membership.items() if incident.user in users
        )
        last_present = present_months[-1] if present_months else pd.NaT
        first_absent = (
            next(
                (
                    month
                    for month in sorted(membership)
                    if month > last_present and incident.user not in membership[month]
                ),
                pd.NaT,
            )
            if pd.notna(last_present)
            else pd.NaT
        )
        incident_end_month = incident.end.to_period("M")
        rows.append(
            {
                "details": incident.details,
                "ldap_last_present_month": str(last_present),
                "ldap_first_absent_month": str(first_absent),
                "absence_month_offset_from_incident_end": (
                    int(first_absent.ordinal - incident_end_month.ordinal)
                    if pd.notna(first_absent)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--save-daily", action="store_true")
    args = parser.parse_args()

    root = (
        args.project_root.resolve()
        if args.project_root
        else find_project_root(Path(__file__).parent)
    )
    data_dir = find_data_dir(root)
    answers_dir = root / "Data" / "archive" / "answers"
    output_dir = root / "Code" / "Analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"수업 데이터: {data_dir}")
    # 이 단계에서는 answers를 열지 않는다. 빈 집합은 집계 함수가 정답 ID를
    # 참조하지 않게 하며, 점수 계산은 로그와 사전 정의한 도메인 가중치만 사용한다.
    covered: dict[str, set[str]] = defaultdict(set)

    logon = aggregate_logon(data_dir / "logon.csv", set(), covered)
    print("logon 집계 완료")
    device, device_events = aggregate_device(
        data_dir / "device.csv", set(), covered
    )
    print("device 집계 완료")
    file_features = aggregate_file(
        data_dir / "file.csv", set(), covered, device_events
    )
    del device_events
    print("file 및 USB 구간 결합 완료")
    http = aggregate_http(data_dir / "http.csv", set(), covered)
    print("http 집계 완료")
    email = aggregate_email(data_dir / "email.csv", set(), covered)
    print("email 집계 완료")

    daily = build_daily_grid([logon, device, file_features, http, email])
    all_weights = GENERIC_WEIGHTS | SEMANTIC_WEIGHTS
    scored = add_z_scores(daily, list(all_weights))
    scored["generic_score"] = weighted_score(scored, GENERIC_WEIGHTS)
    scored["hybrid_score"] = weighted_score(scored, all_weights)
    ranking = build_user_ranking(scored)
    ranking["top_reasons"] = explain_peaks(scored, ranking, all_weights)

    # 여기까지가 정답 비공개 단계다. 이 파일이 확정된 후에만 answers를 연다.
    blind_review = build_blind_review(data_dir, ranking)
    ranking.to_csv(output_dir / "blind_user_risk_ranking.csv", index=False)
    blind_review.to_csv(output_dir / "blind_top50_context_review.csv", index=False)
    blind_manifest = {
        "generic_weights": GENERIC_WEIGHTS,
        "semantic_weights": SEMANTIC_WEIGHTS,
        "rolling_days": ROLLING_DAYS,
        "minimum_baseline_days": MIN_BASELINE_DAYS,
        "zero_std_z": ZERO_STD_Z,
        "z_cap": Z_CAP,
        "note": "이 설정과 순위는 answers를 읽기 전에 확정됨",
    }
    (output_dir / "blind_scoring_manifest.json").write_text(
        json.dumps(blind_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("정답 비공개 순위 확정 완료 — 이제 answers로 사후 채점합니다.")
    insiders, answer_ids = load_answers(answers_dir)
    covered = scan_answer_coverage(data_dir, answer_ids)
    evaluated, metrics, scenarios = evaluate(
        insiders, ranking, scored, all_weights
    )
    coverage = answer_coverage_frame(answer_ids, covered)
    ldap = analyze_ldap(data_dir, insiders)
    evaluated = evaluated.merge(ldap, on="details", how="left")

    evaluated.to_csv(output_dir / "insider_detection.csv", index=False)
    metrics.to_csv(output_dir / "metrics_at_k.csv", index=False)
    scenarios.to_csv(output_dir / "scenario_metrics.csv", index=False)
    coverage.to_csv(output_dir / "answer_sampling_coverage.csv", index=False)
    scored.nlargest(200, "hybrid_score").to_csv(
        output_dir / "top_suspicious_days.csv", index=False
    )
    if args.save_daily:
        scored.to_csv(output_dir / "daily_scores.csv", index=False)

    summary = {
        "data_dir": str(data_dir),
        "date_start": str(daily["event_date"].min().date()),
        "date_end": str(daily["event_date"].max().date()),
        "users": int(daily["user"].nunique()),
        "user_days": int(len(daily)),
        "insiders": int(insiders["user"].nunique()),
        "rolling_days": ROLLING_DAYS,
        "minimum_baseline_days": MIN_BASELINE_DAYS,
        "zero_std_z": ZERO_STD_Z,
        "z_cap": Z_CAP,
        "peak_in_answer_period": int(
            evaluated["overall_peak_in_answer_period"].sum()
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
