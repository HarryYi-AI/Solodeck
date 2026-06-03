from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


CONTENT_DEFAULTS = {
    "body": "",
    "tags": "",
    "language": "",
    "cover_style": "default",
    "followers_before": 0,
    "impressions": 0,
    "completion_rate": 0.0,
    "cost": 0.0,
    "ad_spend": 0.0,
    "is_sponsored": False,
    "series_id": "",
    "parent_content_id": "",
    "content_similarity_group": "",
    "knowledge_domain": "",
    "difficulty_level": "",
    "novelty_score": 0.0,
    "duplication_risk": 0.0,
    "user_fatigue_risk": 0.0,
}


def standardize_contents(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, value in CONTENT_DEFAULTS.items():
        if col not in out.columns:
            out[col] = value
    if "publish_time" in out.columns:
        out["publish_time"] = pd.to_datetime(out["publish_time"])
    if "impressions" in out.columns and "views" in out.columns:
        out["impressions"] = out["impressions"].fillna(0)
        missing_impressions = out["impressions"].eq(0)
        out.loc[missing_impressions, "impressions"] = (out.loc[missing_impressions, "views"] * 1.6).astype(int)
    if "cost" in out.columns:
        out["cost"] = out["cost"].fillna(0)
    if "language" in out.columns:
        out["language"] = out["language"].fillna("")
    return out


def load_contents(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "mock_contents.csv")
    return standardize_contents(df)


def load_revenues(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "mock_revenues.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_campaigns(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "mock_campaigns.csv")
    df["deadline"] = pd.to_datetime(df["deadline"]).dt.date
    return df


def load_ab_tests(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "mock_ab_tests.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_products(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path or DATA_DIR / "mock_products.csv")
    if not target.exists():
        return pd.DataFrame()
    df = pd.read_csv(target)
    df["launch_date"] = pd.to_datetime(df["launch_date"]).dt.date
    return df


def load_feedback(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path or DATA_DIR / "mock_feedback.csv")
    if not target.exists():
        return pd.DataFrame()
    df = pd.read_csv(target)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def load_beta_tests(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path or DATA_DIR / "mock_beta_tests.csv")
    if not target.exists():
        return pd.DataFrame()
    df = pd.read_csv(target)
    df["invited_at"] = pd.to_datetime(df["invited_at"], errors="coerce")
    df["experienced_at"] = pd.to_datetime(df["experienced_at"], errors="coerce")
    return df


def load_all(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir) if data_dir else DATA_DIR
    return (
        load_contents(base / "mock_contents.csv"),
        load_revenues(base / "mock_revenues.csv"),
        load_campaigns(base / "mock_campaigns.csv"),
        load_ab_tests(base / "mock_ab_tests.csv"),
        load_products(base / "mock_products.csv"),
        load_feedback(base / "mock_feedback.csv"),
        load_beta_tests(base / "mock_beta_tests.csv"),
    )


def read_uploaded_csv(uploaded_file, date_columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    for col in date_columns or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df
