import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from simple_quant import config


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = Path(db_path or config.DB_PATH)
    ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_registry (
                dataset_name TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                description TEXT,
                row_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_name TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                target_col TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                features_json TEXT NOT NULL,
                threshold REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS score_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                target_col TEXT NOT NULL,
                probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                as_of_date TEXT NOT NULL,
                scored_at TEXT NOT NULL
            )
            """
        )


def dataset_table_name(dataset_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in dataset_name.lower())
    return f"dataset_{safe}"


def save_dataset(
    df: pd.DataFrame,
    dataset_name: str = config.DEFAULT_DATASET_NAME,
    description: str = "",
    db_path: Optional[Path] = None,
    append: bool = False, 
) -> str:
    init_db(db_path)
    table_name = dataset_table_name(dataset_name)
    now = datetime.now().isoformat()

    payload = df.copy()
    #for column in payload.columns:
    #    if pd.api.types.is_datetime64_any_dtype(payload[column]):
    #        payload[column] = payload[column].dt.strftime("%Y-%m-%d")

    # NOVO: Define o comportamento do pandas to_sql
    sql_behavior = "append" if append else "replace"

    with get_connection(db_path) as conn:
        # NOVO: Passa a variável sql_behavior para o if_exists
        payload.to_sql(table_name, conn, if_exists=sql_behavior, index=False)
        
        # NOVO: Garante que o row_count no registry seja o total real da tabela, 
        # e não apenas o tamanho do chunk que sofreu append.
        total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        conn.execute(
            """
            INSERT INTO dataset_registry (dataset_name, table_name, description, row_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_name) DO UPDATE SET
                table_name=excluded.table_name,
                description=excluded.description,
                row_count=excluded.row_count,
                updated_at=excluded.updated_at
            """,
            (dataset_name, table_name, description, total_rows, now), # Atualizado para total_rows
        )

    return table_name

def load_dataset(
    dataset_name: str = config.DEFAULT_DATASET_NAME,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    init_db(db_path)
    table_name = dataset_table_name(dataset_name)
    with get_connection(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"])
        
    return df


def list_datasets(db_path: Optional[Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql("SELECT * FROM dataset_registry ORDER BY dataset_name", conn)


def save_model_record(
    model_name: str,
    ticker: str,
    target_col: str,
    artifact_path: str,
    features: list[str],
    threshold: Optional[float],
    db_path: Optional[Path] = None,
) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_registry (model_name, ticker, target_col, artifact_path, features_json, threshold, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                ticker=excluded.ticker,
                target_col=excluded.target_col,
                artifact_path=excluded.artifact_path,
                features_json=excluded.features_json,
                threshold=excluded.threshold
            """,
            (
                model_name,
                ticker,
                target_col,
                artifact_path,
                json.dumps(features),
                threshold,
                datetime.now().isoformat(),
            ),
        )


def get_model_record(model_name: str, db_path: Optional[Path] = None) -> dict:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT model_name, ticker, target_col, artifact_path, features_json, threshold, created_at
            FROM model_registry
            WHERE model_name = ?
            """,
            (model_name,),
        ).fetchone()

    if row is None:
        raise ValueError(f"Modelo '{model_name}' não encontrado no banco")

    return {
        "model_name": row[0],
        "ticker": row[1],
        "target_col": row[2],
        "artifact_path": row[3],
        "features": json.loads(row[4]),
        "threshold": row[5],
        "created_at": row[6],
    }


def list_models(db_path: Optional[Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        return pd.read_sql("SELECT * FROM model_registry ORDER BY created_at DESC", conn)


def save_score_row(
    model_name: str,
    ticker: str,
    target_col: str,
    probability: float,
    prediction: int,
    as_of_date: str,
    db_path: Optional[Path] = None,
) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO score_log (model_name, ticker, target_col, probability, prediction, as_of_date, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                ticker,
                target_col,
                float(probability),
                int(prediction),
                as_of_date,
                datetime.now().isoformat(),
            ),
        )


def list_scores(
    model_name: Optional[str] = None,
    ticker: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    init_db(db_path)
    query = "SELECT * FROM score_log WHERE 1=1"
    params = []
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    query += " ORDER BY scored_at DESC"
    with get_connection(db_path) as conn:
        return pd.read_sql(query, conn, params=params)
