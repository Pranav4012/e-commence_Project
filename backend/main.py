from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Delivery Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DATA: dict[str, Any] = {"path": None, "audit": None, "frame": None, "tracking_history": None}
CURRENT_MODEL: dict[str, Any] = {"pipeline": None, "result": None, "features": [], "baseline": {}}
DATASET_LOCK = threading.Lock()
TRACKING_HISTORY_LOCK = threading.Lock()

TRACKING_ALIASES = {
    "order_id": ("order_id", "order_number", "package_id", "tracking_number", "shipment_id"),
    "order_date": ("order_date", "purchase_date", "created_at", "order_timestamp"),
    "delivery_date": ("delivery_date", "actual_delivery_date", "delivered_at"),
    "status": ("order_status", "shipment_status", "delivery_status", "status"),
    "completed": ("delivery_completed", "is_delivered", "completed"),
    "delayed": ("is_delayed", "delayed", "late_delivery"),
    "delay_days": ("delivery_delay_days", "delay_days", "lateness_days"),
    "actual_days": ("actual_delivery_days", "actual_transit_days"),
    "promised_days": ("promised_delivery_days", "estimated_delivery_days", "transit_days"),
    "shipping_service": ("shipping_service", "shipping_mode", "carrier"),
    "fulfilment": ("fulfilment_type", "fulfillment_type", "fulfilment", "fulfillment"),
    "distance": ("estimated_distance_km", "distance_km", "distance"),
    "weather": ("weather_disruption_flag", "weather_disruption", "weather_flag"),
    "carrier_load": ("carrier_load_factor", "carrier_load", "load_factor"),
    "processing_days": ("processing_time_days", "processing_days"),
    "product": ("product_name", "product", "item_name"),
    "category": ("category", "product_category"),
    "city": ("city", "destination_city"),
    "state": ("state", "destination_state"),
}


class TrainRequest(BaseModel):
    target_column: str
    feature_columns: list[str] = Field(min_length=1)
    date_column: str | None = None


class PredictRequest(BaseModel):
    features: dict[str, Any]


class ExplainRequest(BaseModel):
    features: dict[str, Any]


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def load_csv(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("Could not parse CSV. " + " | ".join(errors[-2:]))


def profile_frame(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    rows, cols = df.shape
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    column_profiles = []
    for col in df.columns:
        series = df[col]
        examples = [json_safe(x) for x in series.dropna().head(3).tolist()]
        column_profiles.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
            "examples": examples,
        })

    date_candidates = []
    for col in df.columns:
        name = str(col).lower()
        if any(token in name for token in ("date", "time", "timestamp")):
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.notna().mean()
            if valid >= 0.5:
                date_candidates.append({
                    "column": str(col),
                    "valid_pct": round(float(valid * 100), 2),
                    "min": json_safe(parsed.min()),
                    "max": json_safe(parsed.max()),
                })

    numeric = df.select_dtypes(include=np.number)
    numeric_summary = []
    for col in numeric.columns[:20]:
        s = numeric[col]
        numeric_summary.append({
            "name": str(col),
            "mean": json_safe(round(float(s.mean()), 4)) if s.notna().any() else None,
            "median": json_safe(round(float(s.median()), 4)) if s.notna().any() else None,
            "min": json_safe(s.min()),
            "max": json_safe(s.max()),
        })

    return {
        "filename": filename,
        "rows": int(rows),
        "columns": int(cols),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / max(rows, 1) * 100, 2),
        "total_missing": int(missing.sum()),
        "missing_pct": round(float(missing.sum() / max(rows * cols, 1) * 100), 2),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 ** 2)), 2),
        "date_candidates": date_candidates,
        "columns_profile": column_profiles,
        "numeric_summary": numeric_summary,
        "preview": [
            {str(k): json_safe(v) for k, v in row.items()}
            for row in df.head(8).replace({np.nan: None}).to_dict(orient="records")
        ],
        "warnings": [
            "Column names and target definitions must be verified before model training.",
            "A date-like column is only a candidate; confirm its meaning and availability at order placement.",
            "This audit does not establish that records are original, verified, or independent."
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "delivery-intelligence-api"}


@app.post("/api/audit")
async def audit_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    if file.size and file.size > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the 500 MB upload limit.")
    safe_name = Path(file.filename).name
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        stored.write_bytes(content)
        df = load_csv(stored)
        if df.empty or len(df.columns) == 0:
            raise HTTPException(status_code=400, detail="The CSV contains no usable rows or columns.")
        result = profile_frame(df, safe_name)
        with DATASET_LOCK:
            CURRENT_DATA.update({"path": str(stored), "audit": result, "frame": df, "tracking_history": None})
        CURRENT_MODEL.update({"pipeline": None, "result": None, "features": [], "baseline": {}})
        return result
    except HTTPException:
        stored.unlink(missing_ok=True)
        raise
    except Exception as exc:
        stored.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Dataset audit failed: {exc}") from exc


@app.get("/api/audit/current")
def current_audit():
    ensure_current_dataset()
    if CURRENT_DATA["audit"] is None:
        raise HTTPException(status_code=404, detail="No dataset has been uploaded in this session.")
    return CURRENT_DATA["audit"]


@app.get("/api/analytics")
def analytics():
    ensure_current_dataset()
    df = CURRENT_DATA.get("frame")
    if df is None:
        raise HTTPException(status_code=404, detail="Upload a dataset to calculate analytics.")
    date_cols = [x["column"] for x in CURRENT_DATA["audit"]["date_candidates"]]
    date_col = date_cols[0] if date_cols else None
    time_series = []
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        counts = dates.dropna().dt.to_period("M").value_counts().sort_index()
        time_series = [{"period": str(k), "orders": int(v)} for k, v in counts.items()]

    return {
        "time_series": time_series,
        "date_column_used": date_col,
        "message": "These are descriptive counts only. Delay rates require a verified outcome column and label definition."
    }


def tracking_columns(df: pd.DataFrame) -> dict[str, str | None]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
        for column in df.columns
    }
    return {
        key: next(
            (normalized[re.sub(r"[^a-z0-9]", "", alias.lower())] for alias in aliases
             if re.sub(r"[^a-z0-9]", "", alias.lower()) in normalized),
            None,
        )
        for key, aliases in TRACKING_ALIASES.items()
    }


def tracking_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    return numeric.eq(1) | (numeric.isna() & text.isin(["true", "yes", "y", "delivered", "late", "delayed"]))


def ensure_current_dataset() -> None:
    if CURRENT_DATA["frame"] is not None:
        return
    with DATASET_LOCK:
        if CURRENT_DATA["frame"] is not None:
            return
        available = sorted(UPLOAD_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not available:
            return
        stored = available[0]
        try:
            df = load_csv(stored)
            if df.empty or len(df.columns) == 0:
                raise ValueError("The most recently uploaded CSV has no usable rows or columns.")
            filename = re.sub(r"^[0-9a-f]{32}_", "", stored.name, flags=re.IGNORECASE)
            result = profile_frame(df, filename)
            CURRENT_DATA.update({
                "path": str(stored),
                "audit": result,
                "frame": df,
                "tracking_history": None,
            })
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not restore the most recently uploaded CSV ({stored.name}): {exc}",
            ) from exc


def delivery_completed(df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series:
    if columns["completed"]:
        return tracking_bool(df[columns["completed"]])
    if columns["status"]:
        status = df[columns["status"]].astype("string").str.lower()
        return status.str.contains("deliver", na=False) & ~status.str.contains(
            "return|cancel|reject|lost|damage", na=False
        )
    return pd.Series(False, index=df.index, dtype=bool)


def tracking_delay_flag(df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series:
    if columns["delayed"]:
        return tracking_bool(df[columns["delayed"]])
    if columns["delay_days"]:
        return pd.to_numeric(df[columns["delay_days"]], errors="coerce").gt(0)
    if columns["actual_days"] and columns["promised_days"]:
        actual = pd.to_numeric(df[columns["actual_days"]], errors="coerce")
        promised = pd.to_numeric(df[columns["promised_days"]], errors="coerce")
        return actual.gt(promised)
    return pd.Series(False, index=df.index, dtype=bool)


def wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return {"lower": round(float(max(0, center - margin)), 4), "upper": round(float(min(1, center + margin)), 4)}


def tracking_history(df: pd.DataFrame, columns: dict[str, str | None]) -> dict[str, Any]:
    cached = CURRENT_DATA.get("tracking_history")
    if cached is not None:
        return cached

    completed = delivery_completed(df, columns)
    if columns["actual_days"]:
        days = pd.to_numeric(df[columns["actual_days"]], errors="coerce")
    elif columns["order_date"] and columns["delivery_date"]:
        orders = pd.to_datetime(df[columns["order_date"]], errors="coerce")
        delivered = pd.to_datetime(df[columns["delivery_date"]], errors="coerce")
        days = (delivered - orders).dt.total_seconds() / 86400
    else:
        days = pd.Series(np.nan, index=df.index)

    valid = completed & days.notna() & days.ge(0)
    history = pd.DataFrame({"days": days[valid].astype(float)})
    if columns["shipping_service"]:
        history["shipping_service"] = df.loc[valid, columns["shipping_service"]].astype("string").fillna("")
    if columns["fulfilment"]:
        history["fulfilment"] = df.loc[valid, columns["fulfilment"]].astype("string").fillna("")

    distributions: dict[str, Any] = {"all": np.sort(history["days"].to_numpy())}
    if {"shipping_service", "fulfilment"}.issubset(history.columns):
        distributions["service_fulfilment"] = {
            (str(service), str(fulfilment)): np.sort(group["days"].to_numpy())
            for (service, fulfilment), group in history.groupby(
                ["shipping_service", "fulfilment"], dropna=False, sort=False
            )
        }
    if "shipping_service" in history.columns:
        distributions["service"] = {
            str(service): np.sort(group["days"].to_numpy())
            for service, group in history.groupby("shipping_service", dropna=False, sort=False)
        }

    references: dict[str, dict[str, float]] = {}
    for key in ("carrier_load", "processing_days", "distance"):
        column = columns[key]
        if column:
            values = pd.to_numeric(df.loc[completed, column], errors="coerce").dropna()
            if not values.empty:
                references[key] = {
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                }
    cached = {
        "distributions": distributions,
        "references": references,
        "completed_count": int(completed.sum()),
    }
    CURRENT_DATA["tracking_history"] = cached
    return cached


def safe_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def tracking_list_item(row: pd.Series, columns: dict[str, str | None]) -> dict[str, Any]:
    order_date = pd.to_datetime(row[columns["order_date"]], errors="coerce") if columns["order_date"] else pd.NaT
    completed = bool(tracking_bool(pd.Series([row[columns["completed"]]])).iloc[0]) if columns["completed"] else False
    delayed = bool(tracking_bool(pd.Series([row[columns["delayed"]]])).iloc[0]) if columns["delayed"] else False
    status = safe_text(row[columns["status"]]) if columns["status"] else None
    if columns["completed"] is None and status:
        completed = "deliver" in status.lower() and not any(
            token in status.lower() for token in ("return", "cancel", "reject", "lost", "damage")
        )
    return {
        "order_id": safe_text(row[columns["order_id"]]) if columns["order_id"] else "",
        "order_date": None if pd.isna(order_date) else order_date.strftime("%Y-%m-%d"),
        "status": status or "Status not provided",
        "completed": completed,
        "delayed": delayed if completed else None,
        "product": safe_text(row[columns["product"]]) if columns["product"] else None,
        "destination": " · ".join(
            value for value in [
                safe_text(row[columns["city"]]) if columns["city"] else None,
                safe_text(row[columns["state"]]) if columns["state"] else None,
            ] if value
        ) or None,
        "shipping_service": safe_text(row[columns["shipping_service"]]) if columns["shipping_service"] else None,
    }


@app.get("/api/tracking/analytics")
def tracking_analytics():
    ensure_current_dataset()
    df = CURRENT_DATA.get("frame")
    if df is None:
        raise HTTPException(status_code=404, detail="Upload a dataset before viewing package tracking.")
    columns = tracking_columns(df)
    if columns["order_id"] is None:
        raise HTTPException(status_code=400, detail="The CSV needs an order ID or tracking number column.")

    completed = delivery_completed(df, columns)
    has_delay_outcome = bool(
        (columns["delayed"] and df[columns["delayed"]].notna().any())
        or (columns["delay_days"] and pd.to_numeric(df[columns["delay_days"]], errors="coerce").notna().any())
        or (
            columns["actual_days"] and columns["promised_days"]
            and (
                pd.to_numeric(df[columns["actual_days"]], errors="coerce").notna()
                & pd.to_numeric(df[columns["promised_days"]], errors="coerce").notna()
            ).any()
        )
    )
    delayed = tracking_delay_flag(df, columns) & completed if has_delay_outcome else pd.Series(False, index=df.index)
    total = int(len(df))
    completed_count = int(completed.sum())
    delayed_count = int(delayed.sum())
    status_conflicts = 0
    if columns["status"] and columns["completed"]:
        status_text = df[columns["status"]].astype("string").str.lower()
        status_says_complete = status_text.str.contains("deliver", na=False) & ~status_text.str.contains(
            "return|cancel|reject|lost|damage", na=False
        )
        status_conflicts = int(status_says_complete.ne(completed).sum())
    delivery_dates_on_open = 0
    if columns["delivery_date"]:
        delivery_dates_on_open = int(
            (~completed & pd.to_datetime(df[columns["delivery_date"]], errors="coerce").notna()).sum()
        )
    delay_days = (
        pd.to_numeric(df[columns["delay_days"]], errors="coerce")
        if columns["delay_days"] else pd.Series(np.nan, index=df.index)
    )
    if columns["delay_days"]:
        late_days = delay_days[delayed & delay_days.gt(0)]
    elif columns["actual_days"] and columns["promised_days"]:
        actual_days = pd.to_numeric(df[columns["actual_days"]], errors="coerce")
        promised_days = pd.to_numeric(df[columns["promised_days"]], errors="coerce")
        late_days = (actual_days - promised_days)[delayed]
        late_days = late_days[late_days.gt(0)]
    else:
        late_days = pd.Series(dtype=float)
    status_counts = (
        df[columns["status"]].astype("string").fillna("Status not provided").value_counts().head(12)
        if columns["status"] else pd.Series({"Status not provided": total})
    )

    monthly: list[dict[str, Any]] = []
    if columns["order_date"]:
        dates = pd.to_datetime(df[columns["order_date"]], errors="coerce")
        monthly_frame = pd.DataFrame({
            "month": dates.dt.to_period("M").astype("string"),
            "completed": completed.to_numpy(),
            "delayed": delayed.to_numpy(),
        }).dropna(subset=["month"])
        grouped = monthly_frame.groupby("month", sort=True).agg(
            orders=("month", "size"),
            completed=("completed", "sum"),
            delayed=("delayed", "sum"),
        )
        monthly = [
            {
                "month": str(month),
                "orders": int(item["orders"]),
                "completed": int(item["completed"]),
                "delayed": int(item["delayed"]),
                "delay_rate": round(float(item["delayed"] / item["completed"]), 4)
                if item["completed"] and has_delay_outcome else None,
            }
            for month, item in grouped.iterrows()
        ]

    rate = delayed_count / completed_count if completed_count and has_delay_outcome else None
    delayed_word = "order" if delayed_count == 1 else "orders"
    written_summary = (
        f"{completed_count:,} of {total:,} rows are marked complete in the delivery-completion field. "
        + (
            f"{delayed_count:,} completed {delayed_word} ({rate:.1%}) are marked delayed in the CSV."
            if rate is not None else "The file has no usable delay label or actual-versus-promised duration, so a delay rate cannot be calculated."
        )
    )
    if not columns["status"]:
        written_summary += " No shipment-status column was found."
    if status_conflicts:
        conflict_noun = "row" if status_conflicts == 1 else "rows"
        conflict_verb = "disagree" if status_conflicts != 1 else "disagrees"
        written_summary += f" The status and completion fields {conflict_verb} on {status_conflicts:,} {conflict_noun}; the completion field is used for outcome statistics."
    if delivery_dates_on_open:
        date_noun = "row" if delivery_dates_on_open == 1 else "rows"
        written_summary += f" {delivery_dates_on_open:,} {date_noun} not marked complete still contain a delivery date; those dates are ignored."
    dates = pd.to_datetime(df[columns["order_date"]], errors="coerce") if columns["order_date"] else pd.Series(dtype="datetime64[ns]")
    valid_dates = dates.dropna()
    date_range = {
        "start": valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        "end": valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None,
    }
    return {
        "total_orders": total,
        "completed_orders": completed_count,
        "open_orders": total - completed_count,
        "delayed_completed": delayed_count,
        "delay_rate": round(rate, 4) if rate is not None else None,
        "average_late_days": round(float(late_days.mean()), 2) if not late_days.empty else None,
        "status_completion_conflicts": status_conflicts,
        "delivery_dates_on_open": delivery_dates_on_open,
        "order_date_range": date_range,
        "monthly": monthly,
        "statuses": [{"status": str(status), "count": int(count)} for status, count in status_counts.items()],
        "summary": written_summary,
        "source": "Uploaded CSV snapshot; not live carrier tracking.",
    }


@app.get("/api/orders")
def list_orders(
    search: str = Query(default="", max_length=120),
    status: str = Query(default="", max_length=120),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
):
    ensure_current_dataset()
    df = CURRENT_DATA.get("frame")
    if df is None:
        raise HTTPException(status_code=404, detail="Upload a dataset before looking up packages.")
    columns = tracking_columns(df)
    if columns["order_id"] is None:
        raise HTTPException(status_code=400, detail="The CSV needs an order ID or tracking number column.")
    mask = pd.Series(True, index=df.index)
    if search.strip():
        mask &= df[columns["order_id"]].astype("string").str.contains(
            search.strip(), case=False, na=False, regex=False
        )
    if status.strip() and columns["status"]:
        mask &= df[columns["status"]].astype("string").eq(status.strip()).fillna(False)
    total = int(mask.sum())
    matched_indices = np.flatnonzero(mask.to_numpy())
    start = (page - 1) * page_size
    selected = df.iloc[matched_indices[start:start + page_size]]
    items = [tracking_list_item(row, columns) for _, row in selected.iterrows()]
    statuses = (
        sorted(str(value) for value in df[columns["status"]].dropna().unique())
        if columns["status"] else []
    )
    return {"items": items, "total": total, "page": page, "page_size": page_size, "statuses": statuses}


@app.get("/api/orders/detail")
def order_detail(order_id: str = Query(min_length=1, max_length=120)):
    ensure_current_dataset()
    df = CURRENT_DATA.get("frame")
    if df is None:
        raise HTTPException(status_code=404, detail="Upload a dataset before looking up packages.")
    columns = tracking_columns(df)
    id_column = columns["order_id"]
    if id_column is None:
        raise HTTPException(status_code=400, detail="The CSV needs an order ID or tracking number column.")
    matches = df[id_column].astype("string").eq(order_id)
    if not matches.any():
        raise HTTPException(status_code=404, detail=f"No package with ID {order_id} was found in the uploaded CSV.")
    row = df.loc[matches].iloc[0]
    completed = bool(delivery_completed(df.loc[[row.name]], columns).iloc[0])
    status = safe_text(row[columns["status"]]) if columns["status"] else None
    order_date = pd.to_datetime(row[columns["order_date"]], errors="coerce") if columns["order_date"] else pd.NaT
    csv_delivery_date = pd.to_datetime(row[columns["delivery_date"]], errors="coerce") if columns["delivery_date"] else pd.NaT
    actual_days = pd.to_numeric(pd.Series([row[columns["actual_days"]]]), errors="coerce").iloc[0] if columns["actual_days"] else np.nan
    promised_days = pd.to_numeric(pd.Series([row[columns["promised_days"]]]), errors="coerce").iloc[0] if columns["promised_days"] else np.nan
    history = tracking_history(df, columns)

    service = safe_text(row[columns["shipping_service"]]) if columns["shipping_service"] else None
    fulfilment = safe_text(row[columns["fulfilment"]]) if columns["fulfilment"] else None
    distributions = history["distributions"]
    cohort = distributions.get("service_fulfilment", {}).get((service or "", fulfilment or ""), np.array([]))
    cohort_label = "same shipping service and fulfilment type"
    if len(cohort) < 50 and service:
        cohort = distributions.get("service", {}).get(service, np.array([]))
        cohort_label = "same shipping service"
    if len(cohort) < 50:
        cohort = distributions["all"]
        cohort_label = "all completed deliveries"

    is_delayed = None
    delay_flag_recorded = None
    if not completed and columns["delayed"] and pd.notna(row[columns["delayed"]]):
        delay_flag_recorded = bool(tracking_bool(pd.Series([row[columns["delayed"]]])).iloc[0])
    if completed:
        if columns["delayed"]:
            is_delayed = bool(tracking_bool(pd.Series([row[columns["delayed"]]])).iloc[0])
        elif columns["delay_days"]:
            delay_value = pd.to_numeric(pd.Series([row[columns["delay_days"]]]), errors="coerce").iloc[0]
            is_delayed = bool(delay_value > 0) if pd.notna(delay_value) else None
        elif pd.notna(actual_days) and pd.notna(promised_days):
            is_delayed = bool(actual_days > promised_days)

    predicted_days = float(np.median(cohort)) if len(cohort) else None
    eta_date = None
    promised_date = None
    if not pd.isna(order_date) and pd.notna(promised_days):
        promised_date = (order_date + pd.to_timedelta(int(promised_days), unit="D")).strftime("%Y-%m-%d")
    if completed and not pd.isna(csv_delivery_date):
        eta_date = csv_delivery_date.strftime("%Y-%m-%d")
    elif (
        not completed and predicted_days is not None and not pd.isna(order_date)
        and not any(term in (status or "").lower() for term in ("cancel", "return", "reject", "lost", "damage"))
    ):
        eta_date = (order_date + pd.to_timedelta(int(round(predicted_days)), unit="D")).strftime("%Y-%m-%d")

    estimated_remaining_days = None
    eta_overdue_days = None
    if not completed and eta_date:
        days_to_eta = (pd.Timestamp(eta_date) - pd.Timestamp.now().normalize()).days
        if days_to_eta >= 0:
            estimated_remaining_days = int(days_to_eta)
        else:
            eta_overdue_days = abs(int(days_to_eta))

    arrival_probabilities = []
    probability_by_promised_date = None
    probability_interval = None
    if len(cohort):
        max_day = min(max(int(np.ceil(np.quantile(cohort, 0.95))) + 2, 7), 30)
        arrival_probabilities = [
            {"transit_day": day, "probability": round(float(np.mean(cohort <= day)), 4)}
            for day in range(max_day + 1)
        ]
        if pd.notna(promised_days):
            successes = int(np.sum(cohort <= promised_days))
            probability_by_promised_date = round(float(successes / len(cohort)), 4)
            probability_interval = wilson_interval(successes, len(cohort))

    possible_contributors = []
    weather_column = columns["weather"]
    if weather_column:
        weather_value = pd.to_numeric(pd.Series([row[weather_column]]), errors="coerce").iloc[0]
        if pd.notna(weather_value) and weather_value > 0:
            possible_contributors.append({
                "factor": "Weather disruption",
                "detail": "The CSV sets a weather-disruption flag, but does not identify the event or verify that it caused the delay.",
            })
    for key, label in (
        ("carrier_load", "Carrier load"),
        ("processing_days", "Processing time"),
        ("distance", "Delivery distance"),
    ):
        column = columns[key]
        reference = history["references"].get(key)
        if column and reference:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value) and value >= reference["p75"]:
                unit = " km" if key == "distance" else " days" if key == "processing_days" else ""
                possible_contributors.append({
                    "factor": label,
                    "detail": f"{float(value):g}{unit}, at or above the 75th percentile in completed CSV orders.",
                })
    data_notes = []
    if not completed and not pd.isna(csv_delivery_date):
        data_notes.append("A delivery date is present in the CSV although the completion field is false; it is not treated as an actual delivery.")
    if not possible_contributors:
        possible_contributors.append({
            "factor": "No recorded cause",
            "detail": "The CSV has no explicit event-reason field (such as a landslide) and no thresholded operational signal for this row.",
        })

    delay_days_value = pd.to_numeric(pd.Series([row[columns["delay_days"]]]), errors="coerce").iloc[0] if columns["delay_days"] else np.nan
    timeline = []
    if not pd.isna(order_date):
        timeline.append({"date": order_date.strftime("%Y-%m-%d"), "label": "Order date", "state": "recorded"})
    timeline.append({"date": None, "label": status or "Shipment status not provided", "state": "current"})
    if completed and not pd.isna(csv_delivery_date):
        timeline.append({"date": csv_delivery_date.strftime("%Y-%m-%d"), "label": "Delivery date", "state": "recorded"})
    elif eta_date:
        timeline.append({"date": eta_date, "label": "Estimated arrival (historical cohort)", "state": "estimated"})

    return {
        "order_id": order_id,
        "status": status or "Status not provided",
        "completed": completed,
        "is_delayed": is_delayed,
        "delay_flag_recorded": delay_flag_recorded,
        "order_date": None if pd.isna(order_date) else order_date.strftime("%Y-%m-%d"),
        "actual_delivery_date": csv_delivery_date.strftime("%Y-%m-%d") if completed and not pd.isna(csv_delivery_date) else None,
        "promised_delivery_days": int(promised_days) if pd.notna(promised_days) else None,
        "promised_arrival_date": promised_date,
        "actual_delivery_days": float(actual_days) if completed and pd.notna(actual_days) else None,
        "delay_days": float(delay_days_value) if completed and pd.notna(delay_days_value) else None,
        "estimated_arrival_date": eta_date if not completed else None,
        "estimated_transit_days": round(predicted_days, 1) if predicted_days is not None and not completed else None,
        "estimated_remaining_days": estimated_remaining_days,
        "eta_overdue_days": eta_overdue_days,
        "probability_by_promised_date": probability_by_promised_date if not completed else None,
        "probability_interval": probability_interval if not completed else None,
        "probability_label": (
            "Historical share of completed comparable CSV deliveries arriving by the promised transit day."
        ),
        "arrival_probabilities": arrival_probabilities if not completed else [],
        "cohort_count": int(len(cohort)) if not completed else 0,
        "cohort_description": cohort_label if not completed else None,
        "product": safe_text(row[columns["product"]]) if columns["product"] else None,
        "category": safe_text(row[columns["category"]]) if columns["category"] else None,
        "destination": " · ".join(
            value for value in [
                safe_text(row[columns["city"]]) if columns["city"] else None,
                safe_text(row[columns["state"]]) if columns["state"] else None,
            ] if value
        ) or None,
        "shipping_service": service,
        "fulfilment": fulfilment,
        "possible_contributors": possible_contributors,
        "data_notes": data_notes,
        "timeline": timeline,
        "source": "Uploaded CSV snapshot. Estimates use completed rows in this file and are not live carrier tracking.",
    }


def build_feature_frame(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(df[name]) and not pd.api.types.is_bool_dtype(df[name])]
    categorical = [name for name in features if name not in numeric]
    prepared = df[features].copy()
    for name in numeric:
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    for name in categorical:
        prepared[name] = prepared[name].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return prepared, numeric, categorical


@app.get("/api/model/current")
def current_model():
    if CURRENT_MODEL["result"] is None:
        raise HTTPException(status_code=404, detail="No model has been trained in this session.")
    return CURRENT_MODEL["result"]


@app.post("/api/model/train")
def train_model(request: TrainRequest):
    ensure_current_dataset()
    df = CURRENT_DATA.get("frame")
    if df is None:
        raise HTTPException(status_code=400, detail="Upload a dataset before training a model.")
    CURRENT_MODEL.update({"pipeline": None, "result": None, "features": [], "baseline": {}})
    target = request.target_column
    features = list(dict.fromkeys(request.feature_columns))
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="The selected target column is not in the uploaded dataset.")
    if target in features or any(name not in df.columns for name in features):
        raise HTTPException(status_code=400, detail="Choose existing feature columns, excluding the target.")
    if not features:
        raise HTTPException(status_code=400, detail="Select at least one feature column.")
    date_column = request.date_column
    if date_column is not None:
        if date_column not in df.columns:
            raise HTTPException(status_code=400, detail="The selected date column is not in the uploaded dataset.")
        if date_column == target:
            raise HTTPException(status_code=400, detail="The target column cannot also be used as the chronological split column.")
        if date_column in features:
            raise HTTPException(status_code=400, detail="The chronological split column cannot also be a model feature.")

    working = df[features + [target] + ([date_column] if date_column else [])].copy()
    working[target] = working[target].astype("string").str.strip()
    dropped_missing_target = int((working[target].isna() | working[target].eq("")).sum())
    working = working.dropna(subset=[target])
    working = working[working[target] != ""]
    rows_before_date_filter = len(working)
    if date_column:
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working = working.dropna(subset=[date_column]).sort_values(date_column, kind="stable")
    dropped_missing_date = rows_before_date_filter - len(working)
    if len(working) < 10:
        raise HTTPException(status_code=400, detail="At least 10 rows with a target value are required for a holdout evaluation.")

    labels = working[target].astype(str)
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise HTTPException(status_code=400, detail="The selected target must contain at least two distinct classes.")
    if class_counts.min() < 2:
        raise HTTPException(status_code=400, detail="Each target class needs at least two rows to train and evaluate.")

    X, numeric, categorical = build_feature_frame(working, features)
    y = labels
    if date_column:
        split_at = int(len(working) * 0.8)
        if split_at < 2 or len(working) - split_at < 2:
            raise HTTPException(status_code=400, detail="Not enough rows for the chronological train/test split.")
        X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
        split_method = f"Chronological holdout using {date_column}"
        if y_train.nunique() < 2:
            raise HTTPException(status_code=400, detail="The training period contains fewer than two target classes; adjust the date column or dataset.")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Could not create a stratified holdout: {exc}") from exc
        split_method = "Stratified random holdout (80% train / 20% test)"

    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="(missing)", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    try:
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Model training failed: {exc}") from exc

    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    precision_by_class, recall_by_class, f1_by_class, support_by_class = precision_recall_fscore_support(
        y_test,
        predicted,
        labels=classes,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, predicted, labels=classes).tolist()
    auc = None
    if len(classes) == 2 and y_test.nunique() == 2:
        positive_index = classes.index(classes[1])
        auc = round(float(roc_auc_score(y_test == classes[1], probabilities[:, positive_index])), 4)
    class_metrics = [
        {
            "label": label,
            "precision": round(float(precision_by_class[index]), 4),
            "recall": round(float(recall_by_class[index]), 4),
            "f1": round(float(f1_by_class[index]), 4),
            "support": int(support_by_class[index]),
        }
        for index, label in enumerate(classes)
    ]
    sample_rows = []
    for position, (actual, guess, scores) in enumerate(zip(y_test.iloc[:20], predicted[:20], probabilities[:20])):
        sample_rows.append({
            "row": int(position + 1),
            "actual": str(actual),
            "predicted": str(guess),
            "max_class_probability": round(float(np.max(scores)), 4),
        })

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision_macro": round(float(precision_score(y_test, predicted, labels=classes, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, predicted, labels=classes, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, predicted, labels=classes, average="macro", zero_division=0)), 4),
        "roc_auc": auc,
        "roc_auc_positive_class": classes[1] if auc is not None else None,
    }
    schema = [
        {
            "name": name,
            "kind": "number" if name in numeric else "category",
            "examples": [str(value) for value in df[name].dropna().astype(str).unique()[:5]],
        }
        for name in features
    ]
    result = {
        "target_column": target,
        "features": features,
        "classes": classes,
        "feature_schema": schema,
        "split_method": split_method,
        "rows_used": int(len(working)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dropped_missing_target": dropped_missing_target,
        "dropped_missing_date": int(dropped_missing_date),
        "metrics": metrics,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
        "sample_predictions": sample_rows,
        "model_name": "Random forest classifier",
    }
    transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    encoded_importance = pipeline.named_steps["classifier"].feature_importances_
    feature_importance: dict[str, float] = {name: 0.0 for name in features}
    for encoded_name, importance in zip(transformed_names, encoded_importance):
        source_name = str(encoded_name)
        if source_name.startswith("numeric__"):
            source_name = source_name[len("numeric__"):]
        elif source_name.startswith("categorical__"):
            encoded_field = source_name[len("categorical__"):]
            source_name = next(
                (name for name in sorted(features, key=len, reverse=True)
                 if encoded_field == name or encoded_field.startswith(f"{name}_")),
                "",
            )
        else:
            source_name = ""
        if source_name in feature_importance:
            feature_importance[source_name] += float(importance)
    result["feature_importance"] = [
        {"feature": name, "importance": round(value, 6)}
        for name, value in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    ]
    result["feature_importance_method"] = "Random-forest impurity importance; descriptive, not causal."

    robustness_size = min(500, len(X_test))
    robustness_indices = X_test.sample(n=robustness_size, random_state=42).index
    robustness_X = X_test.loc[robustness_indices].copy()
    robustness_y = y_test.loc[robustness_indices]
    base_predictions = pipeline.predict(robustness_X)
    base_accuracy = float(accuracy_score(robustness_y, base_predictions))
    base_balanced_accuracy = float(balanced_accuracy_score(robustness_y, base_predictions))
    robustness_tests = []
    for feature in features:
        masked_X = robustness_X.copy()
        masked_X[feature] = np.nan
        masked_predictions = pipeline.predict(masked_X)
        robustness_tests.append({
            "feature": feature,
            "accuracy": round(float(accuracy_score(robustness_y, masked_predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(robustness_y, masked_predictions)), 4),
            "balanced_accuracy_delta": round(
                float(balanced_accuracy_score(robustness_y, masked_predictions) - base_balanced_accuracy), 4
            ),
            "prediction_flip_rate": round(float(np.mean(masked_predictions != base_predictions)), 4),
        })
    result["robustness"] = {
        "method": "One-feature-at-a-time missingness stress test",
        "sample_rows": int(robustness_size),
        "baseline_accuracy": round(base_accuracy, 4),
        "baseline_balanced_accuracy": round(base_balanced_accuracy, 4),
        "tests": robustness_tests,
    }

    baseline_values = {}
    for name in features:
        if name in numeric:
            median = X_train[name].median()
            baseline_values[name] = float(median) if pd.notna(median) else 0.0
        else:
            values = X_train[name].dropna().mode()
            baseline_values[name] = str(values.iloc[0]) if not values.empty else "(missing)"
    CURRENT_MODEL.update({
        "pipeline": pipeline,
        "result": result,
        "features": features,
        "baseline": baseline_values,
    })
    return result


def model_input_frame(raw_features: dict[str, Any]) -> pd.DataFrame:
    pipeline = CURRENT_MODEL.get("pipeline")
    result = CURRENT_MODEL.get("result")
    if pipeline is None or result is None:
        raise HTTPException(status_code=400, detail="Train a model before generating predictions or explanations.")
    features = CURRENT_MODEL["features"]
    missing = [name for name in features if name not in raw_features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing input values for: {', '.join(missing)}.")
    frame = pd.DataFrame([{name: raw_features[name] for name in features}])
    numeric = [
        item["name"] for item in result["feature_schema"]
        if item["kind"] == "number"
    ]
    categorical = [name for name in features if name not in numeric]
    for name in numeric:
        try:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Enter a valid number for {name}.") from exc
    for name in categorical:
        frame[name] = frame[name].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return frame


@app.post("/api/model/predict")
def predict_order(request: PredictRequest):
    pipeline = CURRENT_MODEL.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Train a model before generating predictions.")
    frame = model_input_frame(request.features)
    probabilities = pipeline.predict_proba(frame)[0]
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    best_index = int(np.argmax(probabilities))
    return {
        "target_column": CURRENT_MODEL["result"]["target_column"],
        "predicted_class": classes[best_index],
        "max_class_probability": round(float(probabilities[best_index]), 4),
        "probabilities": [
            {"label": label, "probability": round(float(probability), 4)}
            for label, probability in zip(classes, probabilities)
        ],
    }


@app.post("/api/model/explain")
def explain_prediction(request: ExplainRequest):
    pipeline = CURRENT_MODEL.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Train a model before generating explanations.")
    frame = model_input_frame(request.features)
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    probabilities = pipeline.predict_proba(frame)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = classes[predicted_index]
    baseline_probability = float(probabilities[predicted_index])
    explanations = []
    for feature in CURRENT_MODEL["features"]:
        baseline_frame = frame.copy()
        baseline_frame[feature] = CURRENT_MODEL["baseline"][feature]
        baseline_probabilities = pipeline.predict_proba(baseline_frame)[0]
        baseline_class_probability = float(baseline_probabilities[predicted_index])
        explanations.append({
            "feature": feature,
            "value": safe_text(frame.iloc[0][feature]) or "(missing)",
            "baseline_value": safe_text(CURRENT_MODEL["baseline"][feature]) or "(missing)",
            "probability_at_value": round(baseline_probability, 4),
            "probability_at_baseline": round(baseline_class_probability, 4),
            "delta": round(baseline_probability - baseline_class_probability, 4),
        })
    return {
        "target_column": CURRENT_MODEL["result"]["target_column"],
        "predicted_class": predicted_class,
        "predicted_probability": round(baseline_probability, 4),
        "baseline": "Each input is compared independently with the training-set median (numeric) or mode (categorical).",
        "method": "Single-feature baseline substitution; sensitivity analysis, not SHAP or causality.",
        "features": sorted(explanations, key=lambda item: abs(item["delta"]), reverse=True),
    }
