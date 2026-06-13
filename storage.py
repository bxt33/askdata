"""SQLite persistence for datasets, audit events, and analysis tasks."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine

from connectors import import_sql_table
from document_ingest import load_source_as_dataframe


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DB_PATH = APP_DIR / "storage" / "shuxi.db"


class AppStorage:
    def __init__(self, db_path: Path | str = DB_PATH, data_dir: Path | str = DATA_DIR) -> None:
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.warehouse_url = os.getenv("WAREHOUSE_DATABASE_URL", "")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists datasets (
                    id text primary key,
                    name text not null,
                    path text not null,
                    table_name text,
                    content_type text not null default 'table',
                    extraction_status text not null default 'parsed',
                    text_preview text not null default '',
                    metadata_json text not null default '{}',
                    rows integer not null,
                    columns integer not null,
                    source_type text not null,
                    created_at real not null
                );

                create table if not exists tasks (
                    id text primary key,
                    query text not null,
                    dataset_id text,
                    status text not null,
                    progress integer not null default 0,
                    result_json text,
                    error text,
                    created_at real not null,
                    updated_at real not null
                );

                create table if not exists audit_events (
                    id integer primary key autoincrement,
                    trace_id text not null,
                    status text not null,
                    risk_level text not null,
                    risk_factors_json text not null,
                    intent text not null,
                    executed_tools_json text not null,
                    result_keys_json text not null,
                    error text,
                    duration_ms integer not null default 0,
                    ts real not null
                );

                create table if not exists users (
                    id text primary key,
                    username text not null unique,
                    password_hash text not null,
                    role text not null,
                    created_at real not null
                );

                create table if not exists sessions (
                    token text primary key,
                    username text not null,
                    expires_at real not null,
                    created_at real not null
                );

                create table if not exists field_change_requests (
                    id text primary key,
                    dataset_id text not null,
                    old_name text not null,
                    new_name text not null,
                    requester text not null,
                    status text not null,
                    created_at real not null
                );

                create table if not exists value_change_requests (
                    id text primary key,
                    dataset_id text not null,
                    column_name text not null,
                    new_value_json text not null,
                    requester text not null,
                    status text not null,
                    created_at real not null
                );
                """
            )
            self._ensure_column(conn, "datasets", "table_name", "text")
            self._ensure_column(conn, "datasets", "content_type", "text not null default 'table'")
            self._ensure_column(conn, "datasets", "extraction_status", "text not null default 'parsed'")
            self._ensure_column(conn, "datasets", "text_preview", "text not null default ''")
            self._ensure_column(conn, "datasets", "metadata_json", "text not null default '{}'")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        cols = [row["name"] for row in conn.execute(f"pragma table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"alter table {table} add column {column} {ddl}")

    def seed_sample_dataset(self, name: str, source_path: Path) -> str:
        existing = self.find_dataset_by_name(name)
        if existing:
            return existing["id"]
        return self.register_dataset(name=name, source_path=source_path, source_type="sample", copy_file=False)

    def register_dataset(self, name: str, source_path: Path, source_type: str = "upload", copy_file: bool = True) -> str:
        df, metadata = load_source_as_dataframe(source_path)
        dataset_id = "ds-" + uuid.uuid4().hex[:10]
        table_name = self._dataset_table_name(dataset_id)
        target_path = self.data_dir / f"{dataset_id}_{Path(name).name}"
        if copy_file:
            shutil.copyfile(source_path, target_path)
        else:
            target_path = source_path
        with self.connect() as conn:
            self._write_dataset_table(df, table_name, conn)
            conn.execute(
                """
                insert into datasets(
                    id, name, path, table_name, content_type, extraction_status,
                    text_preview, metadata_json, rows, columns, source_type, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    str(target_path),
                    table_name,
                    metadata.get("kind", "table"),
                    metadata.get("status", "parsed"),
                    metadata.get("text_preview", ""),
                    json.dumps({**metadata, "warehouse_url": bool(self.warehouse_url)}, ensure_ascii=False),
                    int(df.shape[0]),
                    int(df.shape[1]),
                    source_type,
                    time.time(),
                ),
            )
        return dataset_id

    def save_uploaded_dataset(self, name: str, content: bytes) -> str:
        tmp_path = self.data_dir / f"upload_{uuid.uuid4().hex[:10]}_{Path(name).name}"
        tmp_path.write_bytes(content)
        return self.register_dataset(name=name, source_path=tmp_path, source_type="upload", copy_file=False)

    def register_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        source_type: str = "database",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        metadata = metadata or {"kind": "table", "status": "parsed", "text_preview": ""}
        dataset_id = "ds-" + uuid.uuid4().hex[:10]
        table_name = self._dataset_table_name(dataset_id)
        backup_path = self.data_dir / f"{dataset_id}_{Path(name).stem}.csv"
        df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        with self.connect() as conn:
            self._write_dataset_table(df, table_name, conn)
            conn.execute(
                """
                insert into datasets(
                    id, name, path, table_name, content_type, extraction_status,
                    text_preview, metadata_json, rows, columns, source_type, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    str(backup_path),
                    table_name,
                    metadata.get("kind", "table"),
                    metadata.get("status", "parsed"),
                    metadata.get("text_preview", ""),
                    json.dumps({**metadata, "warehouse_url": bool(self.warehouse_url)}, ensure_ascii=False),
                    int(df.shape[0]),
                    int(df.shape[1]),
                    source_type,
                    time.time(),
                ),
            )
        return dataset_id

    def import_database_table(self, database_url: str, table_name: str, limit: int = 10000) -> str:
        df = import_sql_table(database_url, table_name, limit=limit)
        return self.register_dataframe(
            name=f"{table_name} @ database",
            df=df,
            source_type="database",
            metadata={"kind": "table", "status": "parsed", "text_preview": "", "database_table": table_name},
        )

    def rename_dataset_column(self, dataset_id: str, old_name: str, new_name: str) -> None:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("数据源不存在。")
        if dataset.get("source_type") == "upload":
            raise PermissionError("上传文件数据源不允许在线修改字段；请修改原文件后重新上传。")
        df = self.read_dataset(dataset_id)
        if old_name not in df.columns:
            raise ValueError(f"字段不存在: {old_name}")
        if new_name in df.columns:
            raise ValueError(f"目标字段已存在: {new_name}")
        df = df.rename(columns={old_name: new_name})
        self.update_dataset_dataframe(dataset_id, df)

    def update_dataset_column_values(self, dataset_id: str, column: str, value: Any) -> int:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("数据源不存在。")
        if dataset.get("source_type") == "upload":
            raise PermissionError("上传文件数据源不允许在线修改数据；请修改原文件后重新上传。")
        df = self.read_dataset(dataset_id)
        if column not in df.columns:
            raise ValueError(f"字段不存在: {column}")
        df[column] = value
        self.update_dataset_dataframe(dataset_id, df)
        return int(len(df))

    def update_dataset_dataframe(self, dataset_id: str, df: pd.DataFrame) -> None:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError("数据源不存在。")
        table_name = dataset.get("table_name")
        if not table_name:
            table_name = self._dataset_table_name(dataset_id)
            with self.connect() as conn:
                conn.execute(
                    "update datasets set table_name = ? where id = ?",
                    (table_name, dataset_id),
                )
        with self.connect() as conn:
            self._write_dataset_table(df, table_name, conn)
            conn.execute(
                "update datasets set rows = ?, columns = ? where id = ?",
                (int(df.shape[0]), int(df.shape[1]), dataset_id),
            )

    def create_field_change_request(self, dataset_id: str, old_name: str, new_name: str, requester: str) -> str:
        request_id = "fcr-" + uuid.uuid4().hex[:10]
        with self.connect() as conn:
            conn.execute(
                """
                insert into field_change_requests(id, dataset_id, old_name, new_name, requester, status, created_at)
                values (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (request_id, dataset_id, old_name, new_name, requester, time.time()),
            )
        return request_id

    def list_field_change_requests(self, status: str | None = None) -> List[Dict[str, Any]]:
        sql = "select * from field_change_requests"
        args: tuple[Any, ...] = ()
        if status:
            sql += " where status = ?"
            args = (status,)
        sql += " order by created_at desc"
        with self.connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def approve_field_change_request(self, request_id: str) -> None:
        with self.connect() as conn:
            row = conn.execute("select * from field_change_requests where id = ?", (request_id,)).fetchone()
        if not row:
            raise ValueError("申请不存在。")
        item = dict(row)
        self.rename_dataset_column(item["dataset_id"], item["old_name"], item["new_name"])
        with self.connect() as conn:
            conn.execute("update field_change_requests set status = 'approved' where id = ?", (request_id,))

    def reject_field_change_request(self, request_id: str) -> None:
        with self.connect() as conn:
            conn.execute("update field_change_requests set status = 'rejected' where id = ?", (request_id,))

    def create_value_change_request(self, dataset_id: str, column: str, value: Any, requester: str) -> str:
        request_id = "vcr-" + uuid.uuid4().hex[:10]
        with self.connect() as conn:
            conn.execute(
                """
                insert into value_change_requests(id, dataset_id, column_name, new_value_json, requester, status, created_at)
                values (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (request_id, dataset_id, column, json.dumps(value, ensure_ascii=False), requester, time.time()),
            )
        return request_id

    def list_value_change_requests(self, status: str | None = None) -> List[Dict[str, Any]]:
        sql = "select * from value_change_requests"
        args: tuple[Any, ...] = ()
        if status:
            sql += " where status = ?"
            args = (status,)
        sql += " order by created_at desc"
        with self.connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["new_value"] = json.loads(item.pop("new_value_json"))
            result.append(item)
        return result

    def approve_value_change_request(self, request_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute("select * from value_change_requests where id = ?", (request_id,)).fetchone()
        if not row:
            raise ValueError("申请不存在。")
        item = dict(row)
        count = self.update_dataset_column_values(
            item["dataset_id"],
            item["column_name"],
            json.loads(item["new_value_json"]),
        )
        with self.connect() as conn:
            conn.execute("update value_change_requests set status = 'approved' where id = ?", (request_id,))
        return count

    def reject_value_change_request(self, request_id: str) -> None:
        with self.connect() as conn:
            conn.execute("update value_change_requests set status = 'rejected' where id = ?", (request_id,))

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from datasets order by created_at desc").fetchall()
        return [dict(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("select * from datasets where id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    def find_dataset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("select * from datasets where name = ? order by created_at desc limit 1", (name,)).fetchone()
        return dict(row) if row else None

    def read_dataset(self, dataset_id: str) -> pd.DataFrame:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        table_name = dataset.get("table_name")
        if table_name:
            metadata = json.loads(dataset.get("metadata_json") or "{}")
            if metadata.get("warehouse_url") and self.warehouse_url:
                engine = create_engine(self.warehouse_url)
                return pd.read_sql_table(table_name, engine)
            with self.connect() as conn:
                return pd.read_sql_query(f'select * from "{table_name}"', conn)
        return pd.read_csv(dataset["path"])

    def _dataset_table_name(self, dataset_id: str) -> str:
        return "dataset_" + dataset_id.replace("-", "_")

    def _write_dataset_table(self, df: pd.DataFrame, table_name: str, sqlite_conn: sqlite3.Connection) -> None:
        if self.warehouse_url:
            engine = create_engine(self.warehouse_url)
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            return
        df.to_sql(table_name, sqlite_conn, if_exists="replace", index=False)

    def create_task(self, query: str, dataset_id: str | None = None) -> str:
        task_id = "task-" + uuid.uuid4().hex[:10]
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                insert into tasks(id, query, dataset_id, status, progress, created_at, updated_at)
                values (?, ?, ?, 'queued', 0, ?, ?)
                """,
                (task_id, query, dataset_id, now, now),
            )
        return task_id

    def update_task(self, task_id: str, status: str, progress: int, result: Any = None, error: str = "") -> None:
        result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
        with self.connect() as conn:
            conn.execute(
                """
                update tasks
                set status = ?, progress = ?, result_json = coalesce(?, result_json), error = ?, updated_at = ?
                where id = ?
                """,
                (status, int(progress), result_json, error, time.time(), task_id),
            )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("select * from tasks where id = ?", (task_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["result"] = json.loads(data["result_json"]) if data.get("result_json") else None
        return data

    def list_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from tasks order by created_at desc limit ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result_json"]) if item.get("result_json") else None
            result.append(item)
        return result

    def record_audit_event(self, event: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into audit_events(
                    trace_id, status, risk_level, risk_factors_json, intent,
                    executed_tools_json, result_keys_json, error, duration_ms, ts
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("trace_id", ""),
                    event.get("status", ""),
                    event.get("risk_level", ""),
                    json.dumps(event.get("risk_factors", []), ensure_ascii=False),
                    event.get("intent", ""),
                    json.dumps(event.get("executed_tools", []), ensure_ascii=False),
                    json.dumps(event.get("result_keys", []), ensure_ascii=False),
                    event.get("error", ""),
                    int(event.get("duration_ms", 0)),
                    float(event.get("ts", time.time())),
                ),
            )

    def list_audit_events(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from audit_events order by ts desc limit ?", (limit,)).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            item["risk_factors"] = json.loads(item.pop("risk_factors_json"))
            item["executed_tools"] = json.loads(item.pop("executed_tools_json"))
            item["result_keys"] = json.loads(item.pop("result_keys_json"))
            events.append(item)
        return list(reversed(events))
