"""Core services for the Shuxi intelligent dataframe analysis console.

The module keeps LLM planning, risk control, auditable execution, quality
review, and report export behind one facade. The implementation is intentionally
conservative: the LLM may create a plan, but all data operations are performed
by registered Python tools with validation and bounded output sizes.
"""

from __future__ import annotations

import ast
import html
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from storage import AppStorage

try:
    from langchain_community.chat_models import ChatTongyi
except Exception:  # pragma: no cover - optional dependency at runtime
    ChatTongyi = None  # type: ignore[assignment]

try:
    from dashscope import Generation
except Exception:  # pragma: no cover - optional dependency at runtime
    Generation = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

JsonDict = Dict[str, Any]
ToolFn = Callable[..., Any]

MAX_TABLE_ROWS = 500
MAX_CHART_POINTS = 200
LARGE_EXPORT_ROWS = 1000
HIGH_RISK_TOOLS = {"export_report", "delete_rows", "update_schema"}
SENSITIVE_FIELD_PATTERNS = (
    "phone",
    "mobile",
    "tel",
    "id_card",
    "idcard",
    "identity",
    "身份证",
    "手机号",
    "电话",
    "salary",
    "薪资",
    "password",
    "密码",
    "bank",
    "银行卡",
    "account",
)


@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    allowed_roles: List[str]
    timeout_sec: float = 5.0
    fn: Optional[ToolFn] = None
    high_risk: bool = False


@dataclass
class AnalysisPlan:
    trace_id: str
    intent: str
    steps: List[JsonDict]
    fields: List[str]
    output_type: str
    export_rows: int = 0
    planner: str = "heuristic"
    warnings: List[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    trace_id: str
    status: str
    risk_level: str
    risk_factors: List[str]
    intent: str
    executed_tools: List[str] = field(default_factory=list)
    result_keys: List[str] = field(default_factory=list)
    error: str = ""
    duration_ms: int = 0
    ts: float = field(default_factory=time.time)


class HeadArgs(BaseModel):
    n: int = Field(default=10, ge=1, le=MAX_TABLE_ROWS)


class ColumnArgs(BaseModel):
    column: str


class ValueCountsArgs(BaseModel):
    column: str
    top: int = Field(default=50, ge=1, le=MAX_CHART_POINTS)


class FilterArgs(BaseModel):
    expr: str = Field(min_length=1, max_length=500)


class GroupAggArgs(BaseModel):
    by: str
    column: str
    agg: str = Field(default="mean")


class TopNArgs(BaseModel):
    n: int = Field(default=5, ge=1, le=MAX_TABLE_ROWS)
    by: Optional[str] = None
    ascending: bool = False


class PlotArgs(BaseModel):
    x: str
    y: str = "count"


class AnomalyArgs(BaseModel):
    column: str
    z_threshold: float = Field(default=3.0, ge=0.1, le=10)
    top: int = Field(default=20, ge=1, le=MAX_TABLE_ROWS)


class CleanMissingArgs(BaseModel):
    strategy: str = Field(default="drop_rows")
    columns: List[str] = Field(default_factory=list)


TOOL_ARG_SCHEMAS = {
    "head": HeadArgs,
    "col_mean": ColumnArgs,
    "value_counts": ValueCountsArgs,
    "safe_filter": FilterArgs,
    "group_agg": GroupAggArgs,
    "top_n": TopNArgs,
    "plot_bar": PlotArgs,
    "plot_line": PlotArgs,
    "plot_scatter": PlotArgs,
    "anomaly_detect": AnomalyArgs,
    "clean_missing": CleanMissingArgs,
}


class ToolRegistry:
    """Governed registry for dataframe tools."""

    def __init__(self) -> None:
        self._registry: Dict[str, ToolMeta] = {}
        self._call_counts: Dict[str, int] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        tools = [
            ToolMeta("df_shape", "Return row and column count.", ["analyst", "admin"], fn=self._tool_df_shape),
            ToolMeta("head", "Return first N rows.", ["analyst", "admin"], fn=self._tool_head),
            ToolMeta("profile", "Return schema, missing-rate, and numeric summary.", ["analyst", "admin"], fn=self._tool_profile),
            ToolMeta("missing_summary", "Return missing value summary by column.", ["analyst", "admin"], fn=self._tool_missing_summary),
            ToolMeta("clean_missing", "Clean missing values with a controlled strategy.", ["analyst", "admin"], fn=self._tool_clean_missing),
            ToolMeta("feature_summary", "Return feature type and distribution summary.", ["analyst", "admin"], fn=self._tool_feature_summary),
            ToolMeta("auto_insights", "Generate deterministic automatic insights.", ["analyst", "admin"], fn=self._tool_auto_insights),
            ToolMeta("describe_numeric", "Return descriptive stats for numeric columns.", ["analyst", "admin"], fn=self._tool_describe_numeric),
            ToolMeta("col_mean", "Calculate mean for a numeric column.", ["analyst", "admin"], fn=self._tool_col_mean),
            ToolMeta("numeric_means", "Calculate means for all numeric columns.", ["analyst", "admin"], fn=self._tool_numeric_means),
            ToolMeta("value_counts", "Count categories in one column.", ["analyst", "admin"], fn=self._tool_value_counts),
            ToolMeta("safe_filter", "Filter rows with a restricted pandas query expression.", ["analyst", "admin"], fn=self._tool_safe_filter),
            ToolMeta("group_agg", "Group by one column and aggregate another column.", ["analyst", "admin"], fn=self._tool_group_agg),
            ToolMeta("top_n", "Sort by a column and return top or bottom rows.", ["analyst", "admin"], fn=self._tool_top_n),
            ToolMeta("correlation", "Return numeric correlation pairs.", ["analyst", "admin"], fn=self._tool_correlation),
            ToolMeta("anomaly_detect", "Detect numeric outliers with z-score.", ["analyst", "admin"], fn=self._tool_anomaly_detect),
            ToolMeta("plot_bar", "Create bar chart payload from current dataframe.", ["analyst", "admin"], fn=self._tool_plot_bar),
            ToolMeta("plot_line", "Create line chart payload from current dataframe.", ["analyst", "admin"], fn=self._tool_plot_line),
            ToolMeta("plot_scatter", "Create scatter chart payload from current dataframe.", ["analyst", "admin"], fn=self._tool_plot_scatter),
            ToolMeta("export_report", "Export full report.", ["admin"], high_risk=True),
            ToolMeta("delete_rows", "Delete dataframe rows.", ["admin"], high_risk=True),
            ToolMeta("update_schema", "Mutate dataframe schema.", ["admin"], high_risk=True),
        ]
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolMeta) -> None:
        self._registry[tool.name] = tool
        self._call_counts.setdefault(tool.name, 0)

    def get(self, name: str) -> Optional[ToolMeta]:
        return self._registry.get(name)

    def list_tools(self, role: str = "analyst") -> List[str]:
        return [name for name, meta in self._registry.items() if role in meta.allowed_roles]

    def metadata(self, role: str = "analyst") -> List[JsonDict]:
        return [
            {"name": name, "description": meta.description, "timeout_sec": meta.timeout_sec}
            for name, meta in self._registry.items()
            if role in meta.allowed_roles
        ]

    def call(self, name: str, role: str, df: pd.DataFrame, args: JsonDict) -> Any:
        meta = self.get(name)
        if meta is None or meta.fn is None:
            raise ValueError(f"Tool is not available: {name}")
        if role not in meta.allowed_roles:
            raise PermissionError(f"Role {role} cannot call tool {name}")
        args = self._validate_args(name, args)
        self._call_counts[name] = self._call_counts.get(name, 0) + 1
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(meta.fn, df, **args)
            try:
                return future.result(timeout=meta.timeout_sec)
            except FutureTimeout as exc:
                raise TimeoutError(f"Tool {name} timed out after {meta.timeout_sec}s") from exc

    def _validate_args(self, name: str, args: JsonDict) -> JsonDict:
        schema = TOOL_ARG_SCHEMAS.get(name)
        if schema is None:
            return args
        try:
            return schema(**args).model_dump()
        except ValidationError as exc:
            raise ValueError(f"工具参数校验失败: {name}: {exc}") from exc

    def call_stats(self) -> Dict[str, int]:
        return dict(self._call_counts)

    @staticmethod
    def _table_from_df(df: pd.DataFrame, limit: int = MAX_TABLE_ROWS) -> JsonDict:
        limited = df.head(limit).copy()
        limited = limited.astype(object).where(pd.notnull(limited), None)
        return {"table": {"columns": [str(c) for c in limited.columns], "data": limited.values.tolist(), "truncated": len(df) > limit}}

    @staticmethod
    def _tool_df_shape(df: pd.DataFrame, **_: Any) -> JsonDict:
        return {"answer": f"数据集共有 {df.shape[0]} 行、{df.shape[1]} 列。"}

    @staticmethod
    def _tool_head(df: pd.DataFrame, n: int = 10, **_: Any) -> pd.DataFrame:
        return df.head(max(1, min(int(n), MAX_TABLE_ROWS)))

    @staticmethod
    def _tool_profile(df: pd.DataFrame, **_: Any) -> JsonDict:
        rows = []
        for col in df.columns:
            rows.append(
                [
                    str(col),
                    str(df[col].dtype),
                    int(df[col].isna().sum()),
                    round(float(df[col].isna().mean()), 4),
                    int(df[col].nunique(dropna=True)),
                ]
            )
        return {"table": {"columns": ["column", "dtype", "missing", "missing_rate", "unique"], "data": rows}}

    @staticmethod
    def _tool_missing_summary(df: pd.DataFrame, **_: Any) -> JsonDict:
        out = pd.DataFrame(
            {
                "column": df.columns.astype(str),
                "missing": df.isna().sum().astype(int).values,
                "missing_rate": df.isna().mean().round(4).values,
            }
        ).sort_values("missing", ascending=False)
        return ToolRegistry._table_from_df(out, limit=MAX_TABLE_ROWS)

    @staticmethod
    def _tool_clean_missing(df: pd.DataFrame, strategy: str = "drop_rows", columns: Optional[List[str]] = None, **_: Any) -> JsonDict:
        columns = columns or list(df.columns)
        _require_columns(df, columns)
        before = int(df.isna().sum().sum())
        cleaned = df.copy()
        if strategy == "drop_rows":
            cleaned = cleaned.dropna(subset=columns)
        elif strategy == "fill_numeric_median":
            for col in columns:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        elif strategy == "fill_text_unknown":
            for col in columns:
                if not pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna("unknown")
        else:
            raise ValueError("strategy must be drop_rows, fill_numeric_median, or fill_text_unknown")
        after = int(cleaned.isna().sum().sum())
        return {
            "answer": f"清洗完成：缺失值从 {before} 个减少到 {after} 个；策略为 {strategy}。",
            "table": ToolRegistry._table_from_df(cleaned.head(50))["table"],
        }

    @staticmethod
    def _tool_feature_summary(df: pd.DataFrame, **_: Any) -> JsonDict:
        rows = []
        for col in df.columns:
            series = df[col]
            row = {
                "column": str(col),
                "dtype": str(series.dtype),
                "missing_rate": round(float(series.isna().mean()), 4),
                "unique": int(series.nunique(dropna=True)),
                "role": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            }
            if pd.api.types.is_numeric_dtype(series):
                row["min"] = round(float(series.min()), 4) if series.notna().any() else None
                row["max"] = round(float(series.max()), 4) if series.notna().any() else None
                row["mean"] = round(float(series.mean()), 4) if series.notna().any() else None
            else:
                row["top"] = str(series.astype("string").fillna("<NA>").mode().iloc[0]) if len(series) else ""
            rows.append(row)
        out = pd.DataFrame(rows)
        return ToolRegistry._table_from_df(out, limit=MAX_TABLE_ROWS)

    @staticmethod
    def _tool_auto_insights(df: pd.DataFrame, **_: Any) -> JsonDict:
        insights = []
        insights.append(f"数据规模为 {df.shape[0]} 行、{df.shape[1]} 列。")
        missing = df.isna().mean().sort_values(ascending=False)
        if len(missing) and missing.iloc[0] > 0:
            insights.append(f"缺失率最高字段是 {missing.index[0]}，缺失率 {missing.iloc[0]:.1%}。")
        numeric = df.select_dtypes(include=["number"])
        if not numeric.empty:
            means = numeric.mean(numeric_only=True).sort_values(ascending=False)
            insights.append(f"均值最高的数值字段是 {means.index[0]}，均值 {means.iloc[0]:,.2f}。")
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True).abs()
            np.fill_diagonal(corr.values, 0)
            max_pair = corr.stack().sort_values(ascending=False)
            if len(max_pair) and max_pair.iloc[0] > 0:
                left, right = max_pair.index[0]
                insights.append(f"相关性最强的字段组合是 {left} 与 {right}，相关系数绝对值 {max_pair.iloc[0]:.3f}。")
        categorical = df.select_dtypes(exclude=["number"])
        if not categorical.empty:
            col = categorical.columns[0]
            top = categorical[col].astype("string").fillna("<NA>").value_counts().head(1)
            if len(top):
                insights.append(f"{col} 中最常见的取值是 {top.index[0]}，出现 {int(top.iloc[0])} 次。")
        return {"answer": "\n".join(f"{i + 1}. {text}" for i, text in enumerate(insights))}

    @staticmethod
    def _tool_describe_numeric(df: pd.DataFrame, **_: Any) -> JsonDict:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            return {"answer": "数据集中没有可用于描述统计的数值列。"}
        out = numeric.describe().T.reset_index().rename(columns={"index": "column"})
        return ToolRegistry._table_from_df(out.round(4), limit=MAX_TABLE_ROWS)

    @staticmethod
    def _tool_col_mean(df: pd.DataFrame, column: str, **_: Any) -> JsonDict:
        _require_columns(df, [column])
        series = pd.to_numeric(df[column], errors="coerce")
        if not series.notna().any():
            return {"answer": f"{column} 列不是可计算均值的数值列。"}
        return {"answer": f"{column} 列的平均值是 {float(series.mean()):,.4f}。"}

    @staticmethod
    def _tool_numeric_means(df: pd.DataFrame, **_: Any) -> JsonDict:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            return {"answer": "数据集中没有数值列。"}
        out = numeric.mean(numeric_only=True).reset_index()
        out.columns = ["column", "mean"]
        return ToolRegistry._table_from_df(out.round(4), limit=MAX_TABLE_ROWS)

    @staticmethod
    def _tool_value_counts(df: pd.DataFrame, column: str, top: int = 50, **_: Any) -> pd.DataFrame:
        _require_columns(df, [column])
        top = max(1, min(int(top), MAX_CHART_POINTS))
        counts = df[column].astype("string").fillna("<NA>").value_counts(dropna=False).head(top)
        return pd.DataFrame({column: counts.index.astype(str), "count": counts.values.astype(int)})

    @staticmethod
    def _tool_safe_filter(df: pd.DataFrame, expr: str, **_: Any) -> pd.DataFrame:
        _validate_query_expr(expr)
        return df.query(expr, engine="python")

    @staticmethod
    def _tool_group_agg(df: pd.DataFrame, by: str, column: str = "", agg: str = "mean", **kwargs: Any) -> pd.DataFrame:
        legacy_agg = kwargs.get("agg")
        if isinstance(legacy_agg, dict) and legacy_agg:
            column, agg = next(iter(legacy_agg.items()))
        if not column:
            raise ValueError("group_agg requires a column argument")
        _require_columns(df, [by, column])
        if agg not in {"mean", "sum", "min", "max", "count", "median"}:
            raise ValueError(f"Unsupported aggregation: {agg}")
        work = df.copy()
        if agg != "count":
            work[column] = pd.to_numeric(work[column], errors="coerce")
        out = work.groupby(by, dropna=False)[column].agg(agg).reset_index()
        out.columns = [by, f"{column}_{agg}"]
        return out.sort_values(out.columns[-1], ascending=False).head(MAX_TABLE_ROWS)

    @staticmethod
    def _tool_top_n(df: pd.DataFrame, n: int = 5, by: Optional[str] = None, ascending: bool = False, **_: Any) -> pd.DataFrame:
        n = max(1, min(int(n), MAX_TABLE_ROWS))
        if by:
            _require_columns(df, [by])
            work = df.copy()
            sort_key = pd.to_numeric(work[by], errors="coerce")
            if sort_key.notna().any():
                work["__sort_key__"] = sort_key
                work = work.sort_values("__sort_key__", ascending=bool(ascending), na_position="last").drop(columns=["__sort_key__"])
            else:
                work = work.sort_values(by, ascending=bool(ascending), na_position="last")
            return work.head(n)
        return df.head(n)

    @staticmethod
    def _tool_correlation(df: pd.DataFrame, **_: Any) -> JsonDict:
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            return {"answer": "至少需要两个数值列才能计算相关性。"}
        corr = numeric.corr(numeric_only=True)
        rows = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    rows.append([str(left), str(right), round(float(value), 4)])
        out = pd.DataFrame(rows, columns=["column_a", "column_b", "correlation"]).sort_values("correlation", key=lambda s: s.abs(), ascending=False)
        return ToolRegistry._table_from_df(out, limit=MAX_TABLE_ROWS)

    @staticmethod
    def _tool_anomaly_detect(df: pd.DataFrame, column: str, z_threshold: float = 3.0, top: int = 20, **_: Any) -> JsonDict:
        _require_columns(df, [column])
        series = pd.to_numeric(df[column], errors="coerce")
        sigma = series.std(ddof=0)
        if not series.notna().any() or sigma == 0 or pd.isna(sigma):
            return {"answer": f"{column} 列无法进行异常检测。"}
        z_score = ((series - series.mean()) / sigma).abs()
        out = df.loc[z_score >= float(z_threshold)].copy()
        out.insert(0, "z_score", z_score.loc[out.index].round(4))
        return ToolRegistry._table_from_df(out.head(max(1, min(int(top), MAX_TABLE_ROWS))), limit=MAX_TABLE_ROWS)

    @staticmethod
    def _tool_plot_bar(df: pd.DataFrame, x: str, y: str = "count", **_: Any) -> JsonDict:
        _require_columns(df, [x])
        if y not in df.columns:
            numeric_cols = list(df.select_dtypes(include=["number"]).columns)
            y = str(numeric_cols[0]) if numeric_cols else str(df.columns[1])
        _require_columns(df, [y])
        out = df[[x, y]].head(MAX_CHART_POINTS).copy()
        return {"bar": {"x": str(x), "y": str(y), "columns": [str(x), str(y)], "data": _records(out), "truncated": len(df) > MAX_CHART_POINTS}}

    @staticmethod
    def _tool_plot_line(df: pd.DataFrame, x: str, y: str, **_: Any) -> JsonDict:
        out = _with_index_column(df, x)
        _require_columns(out, [x, y])
        out = out[[x, y]].head(MAX_CHART_POINTS).copy()
        return {"line": {"x": str(x), "y": str(y), "columns": [str(x), str(y)], "data": _records(out), "truncated": len(df) > MAX_CHART_POINTS}}

    @staticmethod
    def _tool_plot_scatter(df: pd.DataFrame, x: str, y: str, **_: Any) -> JsonDict:
        out = _with_index_column(df, x)
        _require_columns(out, [x, y])
        out = out[[x, y]].head(MAX_CHART_POINTS).copy()
        return {"scatter": {"x": str(x), "y": str(y), "columns": [str(x), str(y)], "data": _records(out), "truncated": len(df) > MAX_CHART_POINTS}}


class PlanEngine:
    """Convert a natural language query into an executable plan."""

    def __init__(
        self,
        registry: ToolRegistry,
        api_key: str = "",
        provider: str = "dashscope",
        model_name: str = "qwen-turbo",
        base_url: str = "",
    ) -> None:
        self.registry = registry
        self.api_key = api_key
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.model = None
        if provider == "dashscope" and api_key and ChatTongyi is not None:
            self.model = ChatTongyi(model=model_name, dashscope_api_key=api_key, temperature=0)

    def parse(self, df_meta: JsonDict, query: str) -> AnalysisPlan:
        warnings: List[str] = []
        if self.model is not None or self._has_provider_client():
            try:
                plan = self._parse_with_llm(df_meta, query)
                plan.planner = "llm"
                return self._sanitize_plan(plan, df_meta, warnings)
            except Exception as exc:
                warnings.append(f"LLM 规划失败，已启用本地规则兜底: {_friendly_llm_error(exc)}")
                logger.warning("LLM planner failed, using heuristic fallback", exc_info=True)
        plan = self._parse_with_heuristics(df_meta, query)
        plan.warnings.extend(warnings)
        return self._sanitize_plan(plan, df_meta, plan.warnings)

    def _parse_with_llm(self, df_meta: JsonDict, query: str) -> AnalysisPlan:
        prompt = self._build_prompt(df_meta, query)
        raw = self._invoke_llm(prompt)
        data = self._safe_parse(str(raw))
        if not data.get("steps"):
            raise ValueError("模型没有返回可执行步骤")
        return AnalysisPlan(
            trace_id="t-" + uuid.uuid4().hex[:10],
            intent=str(data.get("intent") or query),
            steps=list(data.get("steps") or []),
            fields=list(data.get("fields") or []),
            output_type=str(data.get("output_type") or "table"),
            export_rows=int(data.get("export_rows") or 0),
        )

    def _invoke_llm(self, prompt: str) -> str:
        if self.model is not None:
            return str(self.model.invoke(prompt).content)
        if self.provider == "dashscope" and self.api_key and Generation is not None:
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                api_key=self.api_key,
                temperature=0,
                result_format="message",
            )
            output = getattr(response, "output", None) or response.get("output", {})
            if isinstance(output, dict):
                choices = output.get("choices") or []
                if choices:
                    message = choices[0].get("message", {})
                    return str(message.get("content", ""))
                return str(output.get("text", ""))
            return str(getattr(output, "text", ""))
        if self.provider == "deepseek" and self.api_key and OpenAI is not None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url or "https://api.deepseek.com")
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content or ""
        raise RuntimeError(f"未安装或未配置 {self.provider} 模型依赖")

    def _has_provider_client(self) -> bool:
        if self.provider == "dashscope":
            return bool(self.api_key and Generation is not None)
        if self.provider == "deepseek":
            return bool(self.api_key and OpenAI is not None)
        return False

    def _build_prompt(self, df_meta: JsonDict, query: str) -> str:
        return f"""
你是企业级数据分析规划器。只输出 JSON，不要 Markdown，不要解释。

DataFrame:
- shape: {df_meta["shape"]}
- columns: {df_meta["columns"]}
- dtypes: {df_meta["dtypes"]}
- sample: {df_meta["sample_records"]}

可用工具:
{json.dumps(self.registry.metadata(role="analyst"), ensure_ascii=False)}

工具参数:
- df_shape: {{}}
- head: {{"n": 10}}
- profile: {{}}
- missing_summary: {{}}
- describe_numeric: {{}}
- col_mean: {{"column": "列名"}}
- numeric_means: {{}}
- value_counts: {{"column": "列名", "top": 20}}
- safe_filter: {{"expr": "`price` > 5000000 and `area` >= 6000"}}
- group_agg: {{"by": "分组列", "column": "数值列", "agg": "mean|sum|min|max|count|median"}}
- top_n: {{"n": 10, "by": "排序列", "ascending": false}}
- correlation: {{}}
- anomaly_detect: {{"column": "数值列", "z_threshold": 3, "top": 20}}
- plot_bar: {{"x": "列名", "y": "数值列或count"}}
- plot_line: {{"x": "index或日期列", "y": "数值列"}}
- plot_scatter: {{"x": "数值列", "y": "数值列"}}

如果用户要柱状图，通常先 value_counts 或 group_agg，再 plot_bar。
如果用户要趋势且没有日期列，x 使用 index。

用户问题: {query}

输出格式:
{{
  "intent": "一句话概括",
  "steps": [{{"tool": "工具名", "args": {{}}}}],
  "fields": ["涉及字段"],
  "output_type": "answer|table|bar|line|scatter",
  "export_rows": 0
}}
""".strip()

    def _parse_with_heuristics(self, df_meta: JsonDict, query: str) -> AnalysisPlan:
        columns = [str(c) for c in df_meta["columns"]]
        numeric_cols = [c for c, dtype in df_meta["dtypes"].items() if _is_numeric_dtype_name(dtype)]
        q = query.lower()
        mentioned = _mentioned_columns(query, columns)

        def first_numeric(default_index: int = 0) -> str:
            return mentioned[0] if mentioned and mentioned[0] in numeric_cols else (numeric_cols[default_index] if len(numeric_cols) > default_index else columns[0])

        steps: List[JsonDict]
        output_type = "table"
        if any(token in query for token in ("多少行", "几行", "行数", "列数", "shape")):
            steps, output_type = [{"tool": "df_shape", "args": {}}], "answer"
        elif any(token in query for token in ("自动洞察", "洞察", "insight")):
            steps, output_type = [{"tool": "auto_insights", "args": {}}], "answer"
        elif any(token in query for token in ("特征", "字段分析", "feature")):
            steps = [{"tool": "feature_summary", "args": {}}]
        elif any(token in query for token in ("清洗", "处理缺失", "clean")):
            strategy = "fill_numeric_median" if any(token in query for token in ("填充", "中位数")) else "drop_rows"
            steps, output_type = [{"tool": "clean_missing", "args": {"strategy": strategy, "columns": mentioned}}], "table"
        elif any(token in query for token in ("缺失", "空值", "missing")):
            steps = [{"tool": "missing_summary", "args": {}}]
        elif any(token in query for token in ("概况", "质量", "字段", "schema", "profile")):
            steps = [{"tool": "profile", "args": {}}]
        elif any(token in query for token in ("描述统计", "统计摘要", "describe")):
            steps = [{"tool": "describe_numeric", "args": {}}]
        elif any(token in query for token in ("相关", "correlation", "corr")):
            steps = [{"tool": "correlation", "args": {}}]
        elif any(token in query for token in ("异常", "离群", "outlier")):
            steps = [{"tool": "anomaly_detect", "args": {"column": first_numeric(), "z_threshold": 3, "top": 20}}]
        elif any(token in query for token in ("平均", "均值", "mean")) and len(mentioned) <= 1:
            steps, output_type = [{"tool": "col_mean", "args": {"column": first_numeric()}}], "answer"
        elif any(token in query for token in ("所有数值", "数值列均值")):
            steps = [{"tool": "numeric_means", "args": {}}]
        elif any(token in query for token in ("散点", "scatter")) and len(numeric_cols) >= 2:
            x = mentioned[0] if len(mentioned) >= 1 else numeric_cols[0]
            y = mentioned[1] if len(mentioned) >= 2 else numeric_cols[1]
            steps, output_type = [{"tool": "plot_scatter", "args": {"x": x, "y": y}}], "scatter"
        elif any(token in query for token in ("趋势", "折线", "line")):
            y = first_numeric()
            x = _find_date_column(columns) or "index"
            steps, output_type = [{"tool": "plot_line", "args": {"x": x, "y": y}}], "line"
        elif any(token in query for token in ("分布", "占比", "数量", "柱状", "条形", "bar")):
            category = _first_categorical_column(df_meta, mentioned)
            steps = [
                {"tool": "value_counts", "args": {"column": category, "top": 30}},
                {"tool": "plot_bar", "args": {"x": category, "y": "count"}},
            ]
            output_type = "bar"
        elif any(token in query for token in ("最高", "最大", "top", "前")):
            steps = [{"tool": "top_n", "args": {"n": _extract_n(query, 10), "by": first_numeric(), "ascending": False}}]
        elif any(token in query for token in ("最低", "最小", "bottom", "后")):
            steps = [{"tool": "top_n", "args": {"n": _extract_n(query, 10), "by": first_numeric(), "ascending": True}}]
        elif any(token in query for token in ("分组", "按", "group")) and len(mentioned) >= 2:
            by = _first_categorical_column(df_meta, mentioned)
            col = next((c for c in mentioned if c in numeric_cols), first_numeric())
            steps = [{"tool": "group_agg", "args": {"by": by, "column": col, "agg": "mean"}}]
        else:
            steps = [{"tool": "head", "args": {"n": 20}}]

        return AnalysisPlan(
            trace_id="t-" + uuid.uuid4().hex[:10],
            intent=query.strip() or "数据预览",
            steps=steps,
            fields=mentioned,
            output_type=output_type,
            export_rows=0,
            planner="heuristic",
        )

    def _sanitize_plan(self, plan: AnalysisPlan, df_meta: JsonDict, warnings: List[str]) -> AnalysisPlan:
        allowed_tools = set(self.registry.list_tools(role="analyst"))
        columns = set(map(str, df_meta["columns"]))
        sanitized_steps: List[JsonDict] = []
        for step in plan.steps:
            tool = str(step.get("tool", ""))
            if tool not in allowed_tools:
                warnings.append(f"规划中包含不可用工具，已跳过: {tool}")
                continue
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            args = _sanitize_args(tool, args, columns)
            sanitized_steps.append({"tool": tool, "args": args})
        if not sanitized_steps:
            sanitized_steps = [{"tool": "head", "args": {"n": 20}}]
            warnings.append("规划无有效步骤，已回退为数据预览。")
        output_type = plan.output_type if plan.output_type in {"answer", "table", "bar", "line", "scatter"} else "table"
        fields = [field for field in plan.fields if str(field) in columns]
        return AnalysisPlan(
            trace_id=plan.trace_id,
            intent=plan.intent,
            steps=sanitized_steps,
            fields=fields,
            output_type=output_type,
            export_rows=max(0, int(plan.export_rows or 0)),
            planner=plan.planner,
            warnings=warnings,
        )

    @staticmethod
    def _safe_parse(raw: str) -> JsonDict:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
        for candidate in [text, _extract_json_object(text)]:
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                continue
        return {}


class RiskGuard:
    def assess(self, plan: AnalysisPlan, df: Optional[pd.DataFrame] = None) -> Tuple[str, List[str]]:
        factors: List[str] = []
        tools_used = {str(step.get("tool", "")) for step in plan.steps}
        if tools_used & HIGH_RISK_TOOLS:
            factors.append("高危工具: " + ", ".join(sorted(tools_used & HIGH_RISK_TOOLS)))
        sensitive = [field for field in set(plan.fields) if _is_sensitive_field(field)]
        if df is not None:
            sensitive.extend([str(col) for col in df.columns if _is_sensitive_field(str(col)) and str(col) in plan.fields])
        if sensitive:
            factors.append("敏感字段: " + ", ".join(sorted(set(sensitive))))
        if plan.export_rows >= LARGE_EXPORT_ROWS:
            factors.append(f"大批量导出: {plan.export_rows} 行")
        return ("high" if factors else "low"), factors

    def validate_tools(self, plan: AnalysisPlan, registry: ToolRegistry) -> List[str]:
        available = set(registry._registry)
        used = {str(step.get("tool", "")) for step in plan.steps}
        return sorted(used - available)


class AuditLogger:
    def __init__(self, log_path: str | Path = "storage/shuxi.db") -> None:
        self.storage = AppStorage(log_path)
        self._log: List[AuditEvent] = []
        self._load_existing()

    def _load_existing(self) -> None:
        try:
            for item in self.storage.list_audit_events(limit=500):
                self._log.append(
                    AuditEvent(
                        trace_id=item["trace_id"],
                        status=item["status"],
                        risk_level=item["risk_level"],
                        risk_factors=item["risk_factors"],
                        intent=item["intent"],
                        executed_tools=item["executed_tools"],
                        result_keys=item["result_keys"],
                        error=item.get("error") or "",
                        duration_ms=int(item.get("duration_ms") or 0),
                        ts=float(item.get("ts") or time.time()),
                    )
                )
        except Exception:
            logger.warning("Failed to load audit events from sqlite", exc_info=True)

    def record(self, event: AuditEvent) -> None:
        self._log.append(event)
        self.storage.record_audit_event(asdict(event))
        logger.info("audit trace=%s status=%s risk=%s", event.trace_id, event.status, event.risk_level)

    def all_events(self) -> List[JsonDict]:
        return [asdict(event) for event in self._log]

    def recent(self, limit: int = 20) -> List[JsonDict]:
        return self.all_events()[-limit:]


class DataAnalyzer:
    """ReAct execution layer.

    Stage 1 executes registered dataframe tools step by step.
    Stage 2 asks the LLM to convert tool observations into the final JSON
    contract consumed by Streamlit. If no model key is configured, the
    deterministic tool result is returned so the product remains usable.
    """

    def __init__(self, registry: ToolRegistry, llm_caller: Optional[Callable[[str], str]] = None, role: str = "analyst") -> None:
        self.registry = registry
        self.llm_caller = llm_caller
        self.role = role

    def execute(self, plan: AnalysisPlan, df: pd.DataFrame) -> JsonDict:
        work_df = df.copy()
        last_obj: Any = None
        executed: List[str] = []
        observations: List[JsonDict] = []
        for step in plan.steps:
            tool_name = str(step.get("tool", ""))
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            logger.info("execute trace=%s tool=%s args=%s", plan.trace_id, tool_name, args)
            last_obj = self.registry.call(tool_name, self.role, work_df, args)
            executed.append(tool_name)
            if isinstance(last_obj, pd.DataFrame):
                work_df = last_obj
            observations.append({"tool": tool_name, "args": args, "output": self._compact_observation(last_obj)})

        if isinstance(last_obj, dict) and any(key in last_obj for key in ("answer", "table", "bar", "line", "scatter")):
            deterministic_result = dict(last_obj)
        elif isinstance(last_obj, pd.DataFrame):
            deterministic_result = ToolRegistry._table_from_df(last_obj)
        elif isinstance(last_obj, str):
            deterministic_result = {"answer": last_obj}
        else:
            deterministic_result = ToolRegistry._table_from_df(work_df)

        result = self._call_llm_for_result(plan, observations, deterministic_result)

        result["_trace_id"] = plan.trace_id
        result["_executed_tools"] = executed
        result["_planner"] = plan.planner
        if plan.warnings:
            result["_warnings"] = plan.warnings
        return _json_safe(result)

    def _call_llm_for_result(self, plan: AnalysisPlan, observations: List[JsonDict], fallback_result: JsonDict) -> JsonDict:
        """LLM second stage: convert tool observations into final JSON.

        The model is not allowed to invent data. It can only choose or polish
        the already-computed tool result into one of answer/table/bar/line/scatter.
        """
        if self.llm_caller is None:
            result = dict(fallback_result)
            result["_result_stage"] = "tool_direct"
            return result

        prompt = f"""
你是数据分析执行助手。前一步已经通过受控 Pandas 工具完成真实计算。
请基于工具观测结果生成最终 JSON，不要编造任何没有出现在观测结果中的数值。

分析意图: {plan.intent}
期望输出类型: {plan.output_type}
执行计划: {json.dumps(plan.steps, ensure_ascii=False)}
工具观测结果: {json.dumps(observations, ensure_ascii=False)}
兜底结果: {json.dumps(fallback_result, ensure_ascii=False)}

输出协议，只能返回以下结构之一：
- {{"answer": "中文结论"}}
- {{"table": {{"columns": ["列1"], "data": [[值1]]}}}}
- {{"bar": {{"x": "x列", "y": "y列", "columns": ["x列", "y列"], "data": [[x, y]]}}}}
- {{"line": {{"x": "x列", "y": "y列", "columns": ["x列", "y列"], "data": [[x, y]]}}}}
- {{"scatter": {{"x": "x列", "y": "y列", "columns": ["x列", "y列"], "data": [[x, y]]}}}}

只输出 JSON，不要 Markdown，不要解释。
""".strip()
        try:
            raw = self.llm_caller(prompt)
            parsed = PlanEngine._safe_parse(raw)
            if isinstance(parsed, dict) and any(key in parsed for key in ("answer", "table", "bar", "line", "scatter")):
                parsed["_result_stage"] = "llm_second_stage"
                return parsed
        except Exception as exc:
            logger.warning("LLM result stage failed, using tool result", exc_info=True)
            result = dict(fallback_result)
            result["_result_stage"] = "tool_direct"
            result["_warnings"] = [f"LLM 第二阶段失败，已使用工具结果: {_friendly_llm_error(exc)}"]
            return result

        result = dict(fallback_result)
        result["_result_stage"] = "tool_direct"
        result["_warnings"] = ["LLM 第二阶段未返回合法 JSON，已使用工具结果。"]
        return result

    @staticmethod
    def _compact_observation(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return ToolRegistry._table_from_df(value, limit=80)
        if isinstance(value, dict):
            compact = _json_safe(value)
            for key in ("table", "bar", "line", "scatter"):
                if key in compact and isinstance(compact[key], dict):
                    data = compact[key].get("data")
                    if isinstance(data, list) and len(data) > 80:
                        compact[key] = dict(compact[key])
                        compact[key]["data"] = data[:80]
                        compact[key]["truncated"] = True
            return compact
        return _json_safe(value)


class QualityOptimizer:
    def __init__(self, audit: AuditLogger, registry: ToolRegistry) -> None:
        self.audit = audit
        self.registry = registry

    def suggest(self) -> List[str]:
        events = self.audit.all_events()
        if not events:
            return ["暂无审计记录。先完成几次分析后，可以基于失败率、风险拦截和工具命中率给出建议。"]
        failed = [event for event in events if event["status"] == "failed"]
        cancelled = [event for event in events if event["status"] == "cancelled"]
        tips: List[str] = []
        if failed:
            tips.append(f"最近共有 {len(failed)} 次失败，优先检查字段名映射、过滤表达式和模型规划输出。")
        if cancelled:
            tips.append(f"共有 {len(cancelled)} 次高风险拦截，建议为敏感字段建立更细粒度的审批规则。")
        top_tools = sorted(self.registry.call_stats().items(), key=lambda item: item[1], reverse=True)[:3]
        if top_tools:
            tips.append("高频工具: " + "、".join(f"{name}({count})" for name, count in top_tools if count))
        if not tips:
            tips.append("分析链路运行稳定，当前没有明显的失败或风险积压。")
        return tips


class ReportExporter:
    def export_html(self, query: str, result: JsonDict, audit_events: List[JsonDict]) -> str:
        body = [self._result_html(result)]
        recent_audit = audit_events[-20:]
        audit_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(event.get('trace_id', '')))}</td>"
            f"<td>{html.escape(str(event.get('status', '')))}</td>"
            f"<td>{html.escape(str(event.get('risk_level', '')))}</td>"
            f"<td>{html.escape(str(event.get('intent', '')))}</td>"
            "</tr>"
            for event in recent_audit
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>数析智能数据分析报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; overflow: auto; }}
    .meta {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>数析智能数据分析报告</h1>
  <p class="meta">生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
  <p><strong>分析问题:</strong> {html.escape(query)}</p>
  <p><strong>Trace ID:</strong> {html.escape(str(result.get("_trace_id", "N/A")))}</p>
  {''.join(body)}
  <h2>审计日志</h2>
  <table><thead><tr><th>Trace ID</th><th>状态</th><th>风险</th><th>意图</th></tr></thead><tbody>{audit_rows}</tbody></table>
</body>
</html>"""

    def _result_html(self, result: JsonDict) -> str:
        if result.get("blocked"):
            return "<h2>结果</h2><p>高风险操作已阻断。</p>"
        parts = ["<h2>分析结果</h2>"]
        if "answer" in result:
            parts.append(f"<p>{html.escape(str(result['answer']))}</p>")
        if "table" in result:
            parts.append(_html_table(result["table"]))
        for key in ("bar", "line", "scatter"):
            if key in result:
                parts.append(f"<h3>{key} 图表数据</h3><pre>{html.escape(json.dumps(result[key], ensure_ascii=False, indent=2))}</pre>")
        return "".join(parts)


class DataframeAgentFacade:
    def __init__(
        self,
        dashscope_api_key: str = "",
        api_key: str = "",
        provider: str = "dashscope",
        audit_log_path: str | Path = "storage/shuxi.db",
        model_name: str = "qwen-turbo",
        base_url: str = "",
    ) -> None:
        if dashscope_api_key and not api_key:
            api_key = dashscope_api_key
        provider = (provider or os.getenv("LLM_PROVIDER", "dashscope")).lower()
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY", "") if provider == "deepseek" else os.getenv("DASHSCOPE_API_KEY", "")
        if provider == "deepseek" and model_name == "qwen-turbo":
            model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if not base_url and provider == "deepseek":
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.registry = ToolRegistry()
        self.plan_eng = PlanEngine(self.registry, api_key=api_key, provider=provider, model_name=model_name, base_url=base_url)
        self.risk = RiskGuard()
        self.audit = AuditLogger(audit_log_path)
        llm_caller = self.plan_eng._invoke_llm if api_key else None
        self.analyzer = DataAnalyzer(self.registry, llm_caller=llm_caller)
        self.optimizer = QualityOptimizer(self.audit, self.registry)
        self.exporter = ReportExporter()
        self.provider = provider
        self.model_name = model_name

    def analyze(self, df: pd.DataFrame, query: str, human_confirmed: bool = False) -> JsonDict:
        if df is None or df.empty:
            raise ValueError("数据为空，请上传包含数据的 CSV 文件。")
        if not query or not query.strip():
            raise ValueError("分析问题不能为空。")

        start = time.perf_counter()
        df_meta = self._build_df_meta(df)
        plan = self.plan_eng.parse(df_meta, query.strip())
        risk_level, factors = self.risk.assess(plan, df)

        if risk_level == "high" and not human_confirmed:
            event = AuditEvent(plan.trace_id, "cancelled", risk_level, factors, plan.intent)
            self.audit.record(event)
            return {
                "blocked": True,
                "_trace_id": plan.trace_id,
                "reason": "high_risk_not_confirmed",
                "risk_factors": factors,
                "plan": asdict(plan),
            }

        try:
            result = self.analyzer.execute(plan, df)
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.audit.record(
                AuditEvent(
                    trace_id=plan.trace_id,
                    status="executed",
                    risk_level=risk_level,
                    risk_factors=factors,
                    intent=plan.intent,
                    executed_tools=result.get("_executed_tools", []),
                    result_keys=[key for key in result.keys() if not key.startswith("_")],
                    duration_ms=duration_ms,
                )
            )
            return result
        except Exception as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.audit.record(AuditEvent(plan.trace_id, "failed", risk_level, factors, plan.intent, error=str(exc), duration_ms=duration_ms))
            raise ValueError(f"执行分析时出错: {exc}") from exc

    def export_report(self, query: str, result: JsonDict) -> str:
        return self.exporter.export_html(query, result, self.audit.all_events())

    def quality_suggestions(self) -> List[str]:
        return self.optimizer.suggest()

    def _build_df_meta(self, df: pd.DataFrame) -> JsonDict:
        sample = df.head(15).astype(object).where(pd.notnull(df.head(15)), None)
        return {
            "shape": tuple(df.shape),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "sample_records": sample.to_dict(orient="records"),
        }


def dataframe_agent(dashscope_api_key: str, df: pd.DataFrame, query: str) -> JsonDict:
    return DataframeAgentFacade(dashscope_api_key).analyze(df, query, human_confirmed=True)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"列不存在: {', '.join(map(str, missing))}")


def _records(df: pd.DataFrame) -> List[List[Any]]:
    clean = df.astype(object).where(pd.notnull(df), None)
    return clean.values.tolist()


def _with_index_column(df: pd.DataFrame, x: str) -> pd.DataFrame:
    if x != "index":
        return df
    out = df.copy()
    out["index"] = list(range(len(out)))
    return out


def _is_numeric_dtype_name(dtype: str) -> bool:
    return any(token in str(dtype).lower() for token in ("int", "float", "double", "decimal"))


def _mentioned_columns(query: str, columns: List[str]) -> List[str]:
    normalized = query.lower()
    matches = []
    for col in columns:
        lower_col = col.lower()
        pos = normalized.find(lower_col)
        if pos == -1:
            pos = query.find(str(col))
        if pos != -1:
            matches.append((pos, col))
    return [col for _, col in sorted(matches, key=lambda item: item[0])]


def _find_date_column(columns: List[str]) -> Optional[str]:
    for col in columns:
        if any(token in col.lower() for token in ("date", "time", "day", "month", "日期", "时间")):
            return col
    return None


def _first_categorical_column(df_meta: JsonDict, mentioned: List[str]) -> str:
    dtypes = df_meta["dtypes"]
    for col in mentioned:
        if not _is_numeric_dtype_name(dtypes.get(col, "")):
            return col
    for col, dtype in dtypes.items():
        if not _is_numeric_dtype_name(dtype):
            return col
    return str(df_meta["columns"][0])


def _extract_n(query: str, default: int) -> int:
    match = re.search(r"(\d+)", query)
    if not match:
        return default
    return max(1, min(int(match.group(1)), MAX_TABLE_ROWS))


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _sanitize_args(tool: str, args: JsonDict, columns: set[str]) -> JsonDict:
    cleaned = dict(args)
    for key in ("column", "by", "x", "y"):
        if key in cleaned and cleaned[key] not in columns and not (key == "x" and cleaned[key] == "index"):
            cleaned.pop(key)
    if tool in {"col_mean", "value_counts", "anomaly_detect"} and "column" not in cleaned:
        cleaned["column"] = next(iter(columns))
    if tool == "group_agg":
        cleaned.setdefault("by", next(iter(columns)))
        cleaned.setdefault("column", next(iter(columns)))
        cleaned.setdefault("agg", "mean")
    if tool in {"plot_line", "plot_scatter"}:
        ordered = list(columns)
        cleaned.setdefault("x", "index")
        cleaned.setdefault("y", ordered[0])
    if tool == "plot_bar":
        cleaned.setdefault("x", next(iter(columns)))
        cleaned.setdefault("y", "count")
    if tool == "safe_filter" and "expr" in cleaned:
        _validate_query_expr(str(cleaned["expr"]))
    return cleaned


def _validate_query_expr(expr: str) -> None:
    if len(expr) > 500:
        raise ValueError("过滤表达式过长。")
    lowered = expr.lower()
    forbidden = ["__", "import", "exec", "eval", "open(", "read_", "to_", "os.", "sys.", "subprocess", "lambda"]
    if any(token in lowered for token in forbidden):
        raise ValueError("过滤表达式包含不允许的内容。")
    try:
        ast.parse(expr.replace("`", ""), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"过滤表达式语法错误: {expr}") from exc


def _is_sensitive_field(field: str) -> bool:
    normalized = field.lower().replace(" ", "_")
    return any(pattern in normalized for pattern in SENSITIVE_FIELD_PATTERNS)


def _friendly_llm_error(exc: Exception) -> str:
    raw = str(exc)
    lowered = raw.lower()
    if "authentication fails" in lowered or "authentication_error" in lowered or "invalid_request_error" in lowered:
        return (
            "模型服务返回鉴权失败。请检查当前选择的服务商是否和 API Key 匹配，"
            "DeepSeek 请使用 DeepSeek 控制台生成的完整 Key，并确认 Key 未删除、未禁用、未复制缺字符。"
        )
    if "invalidapikey" in lowered or "invalid api-key" in lowered or "status_code: 401" in lowered or "status code: 401" in lowered:
        return (
            "模型服务返回 401 Invalid API Key。请确认当前服务商和 API Key 匹配，"
            "并确认 Key 未禁用、未复制缺字符、账号已开通对应模型服务。"
        )
    if "quota" in lowered or "insufficient" in lowered:
        return "DashScope 额度不足或账户不可用，请检查余额、免费额度或调用权限。"
    if "timeout" in lowered or "timed out" in lowered:
        return "DashScope 请求超时，请稍后重试或检查网络。"
    return raw


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def _html_table(payload: JsonDict) -> str:
    columns = payload.get("columns", [])
    rows = payload.get("data", [])
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    note = "<p class='meta'>表格已截断。</p>" if payload.get("truncated") else ""
    return f"{note}<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
