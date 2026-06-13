"""Database connector utilities for importing external SQL tables."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import create_engine, inspect, text


def list_sql_tables(database_url: str) -> List[str]:
    engine = create_engine(database_url)
    inspector = inspect(engine)
    return inspector.get_table_names()


def preview_sql_table(database_url: str, table_name: str, limit: int = 100) -> pd.DataFrame:
    engine = create_engine(database_url)
    safe_limit = max(1, min(int(limit), 1000))
    with engine.connect() as conn:
        return pd.read_sql_query(text(f'select * from "{table_name}" limit {safe_limit}'), conn)


def import_sql_table(database_url: str, table_name: str, limit: int = 10000) -> pd.DataFrame:
    engine = create_engine(database_url)
    safe_limit = max(1, min(int(limit), 100000))
    with engine.connect() as conn:
        return pd.read_sql_query(text(f'select * from "{table_name}" limit {safe_limit}'), conn)


def connection_summary(database_url: str) -> Dict[str, Any]:
    tables = list_sql_tables(database_url)
    return {"table_count": len(tables), "tables": tables[:50]}
