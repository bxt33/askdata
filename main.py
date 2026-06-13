"""Streamlit UI for the Shuxi intelligent dataframe analysis console."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from auth import AuthService, require_role
from connectors import list_sql_tables
from storage import AppStorage
from utils import DataframeAgentFacade


APP_DIR = Path(__file__).resolve().parent
SAMPLE_FILES = {
    "房价数据 house_price.csv": APP_DIR / "house_price.csv",
    "个人消费 personal_data.csv": APP_DIR / "personal_data.csv",
}
storage = AppStorage(APP_DIR / "storage" / "shuxi.db", APP_DIR / "data")
auth = AuthService(storage)
for sample_label, sample_path in SAMPLE_FILES.items():
    if sample_path.exists():
        storage.seed_sample_dataset(sample_path.name, sample_path)


st.set_page_config(page_title="Shuxi", page_icon="📊", layout="wide")


I18N = {
    "zh": {
        "app_title": "「数析」智能数据分析台",
        "app_caption": "支持 CSV / PDF / Word 上传入库分析，提供结构化规划、受控执行、风险拦截、审计留痕和报告导出。",
        "rows": "行数",
        "cols": "列数",
        "missing": "缺失值",
        "numeric_cols": "数值列",
        "data_preview": "数据预览",
        "field_info": "字段信息",
        "field": "字段",
        "type": "类型",
        "missing_count": "缺失数",
        "unique": "唯一值",
        "empty_chart": "图表数据为空。",
        "chart_truncated": "图表点数较多，已按上限截断展示。",
        "blocked": "高风险操作已被阻断。请确认风险因素后，在侧边栏勾选人工确认再执行。",
        "plan_detail": "计划详情",
        "tools": "工具",
        "planner": "规划器",
        "none": "无",
        "result_truncated": "结果行数较多，已截断展示。",
        "bar": "条形图",
        "line": "折线图",
        "scatter": "散点图",
        "raw_json": "原始 JSON 结果",
        "recent_n": "最近 {n} 条",
        "no_audit": "暂无审计记录",
        "executed": "通过",
        "cancelled": "阻断",
        "failed": "失败",
        "login": "登录",
        "username": "用户名",
        "password": "密码",
        "login_failed": "用户名或密码错误。默认账号：admin / admin123",
        "login_required": "请先登录。默认管理员账号：admin / admin123",
        "admin": "管理员",
        "analyst": "分析员",
        "viewer": "访客",
        "no_dataset_for_field": "当前没有选中的数据源，不能修改字段。",
        "file_field_forbidden": "上传文件数据源不允许在线改字段。样例数据源和数据库导入数据源可以修改。",
        "field_rename_hint": "请明确字段修改，例如：把 price 字段改成 total_price。",
        "field_synced": "字段已同步数据库：{old} -> {new}",
        "field_review": "字段修改需要管理员审核，已创建申请：{id}",
        "value_change_hint": "请明确字段和值，例如：把 category 字段全部改为 1。",
        "value_synced": "字段值已同步数据库：{column} 全部更新为 {value}，共 {count} 条记录。",
        "value_review": "数据值修改需要管理员审核，已创建申请：{id}",
        "pending_value": "待审核数据值修改",
        "value_request_line": "{user} 请求：{column} 全部更新为 {value} ({dataset})",
        "current_user": "当前用户：{username} · {role}",
        "logout": "退出登录",
        "language": "界面语言",
        "config": "配置",
        "provider": "模型服务商",
        "model": "模型名称",
        "api_ok": "已填写 {provider} API Key，将优先调用 {model}；如果服务端报错，会自动切回本地规则。",
        "api_empty": "未填写 API Key，当前使用本地规则规划。",
        "human_confirm": "人工确认高风险操作",
        "audit": "审计日志",
        "quality": "生成质量建议",
        "data_sources": "数据源管理",
        "data_sources_caption": "可以从数据库数据源中选择，也可以上传 CSV、PDF、Word 文档。PDF 文本版会直接解析；扫描件会标记为需要 OCR。",
        "source_select": "选择数据来源",
        "source_library": "数据源库",
        "source_upload": "上传文件（CSV / PDF / Word）",
        "source_db": "导入数据库表",
        "no_source": "暂无数据源，请上传 CSV、PDF 或 Word 文件。",
        "choose_source": "选择数据源",
        "needs_ocr": "这个 PDF 没有可读取文本层，可能是扫描件。当前已标记为需要 OCR，后续可接入 Tesseract/EasyOCR/PaddleOCR。",
        "doc_preview": "文档文本预览",
        "db_url": "数据库连接串",
        "max_rows": "最多导入行数",
        "load_tables": "读取表列表",
        "choose_table": "选择数据库表",
        "import_source": "导入为数据源",
        "imported_table": "已导入数据库表：{table}",
        "db_import_failed": "数据库导入失败：{error}",
        "upload_file": "上传数据文件（支持 .csv / .pdf / .docx / .doc）",
        "upload_help": "CSV 会作为结构化表入库；PDF/DOCX 会抽取文本并转换为 field/value/confidence 字段表。",
        "saved_source": "已保存到数据库数据源：{name}",
        "upload_ocr": "上传成功，但 PDF 可能是扫描件，未识别到文本层。需要接入 OCR 后才能抽取字段。",
        "upload_failed": "上传解析失败：{error}",
        "field_governance": "字段治理",
        "old_field": "原字段",
        "new_field": "新字段名",
        "submit_field": "提交字段修改",
        "pending_field": "待审核字段修改",
        "request_line": "{user} 请求：{old} -> {new} ({dataset})",
        "approve": "批准",
        "reject": "拒绝",
        "approved": "已批准并同步数据库。",
        "rejected": "已拒绝。",
        "tasks": "任务状态",
        "task": "任务",
        "status": "状态",
        "progress": "进度",
        "question": "问题",
        "no_tasks": "暂无任务",
        "reports": "报告中心",
        "download_report_task": "下载任务报告 {id}",
        "no_reports": "暂无可下载报告。",
        "users": "用户与权限",
        "new_username": "新用户名",
        "new_password": "新用户密码",
        "role": "角色",
        "create_user": "创建用户",
        "user_created": "用户已创建。",
        "create_failed": "创建失败：{error}",
        "query": "智能查询",
        "examples": ["展示数据质量概况", "price 最高的前 10 条记录", "furnishingstatus 的分布柱状图", "用散点图展示 area 和 price 的关系", "计算所有数值列的相关性"],
        "input_query": "输入分析问题",
        "example_prefix": "例如：",
        "quick_example": "快速示例",
        "run": "开始分析",
        "choose_first": "请先选择数据库数据源，或上传 CSV / PDF / Word 文件。",
        "query_empty": "请输入分析问题。",
        "analyzing": "正在规划并执行分析...",
        "analysis_failed": "分析失败：{error}",
        "result": "分析结果",
        "task_id": "任务 ID: `{id}`",
        "download_html": "下载 HTML 报告",
    },
    "en": {
        "app_title": "Shuxi Intelligent Data Analysis Console",
        "app_caption": "Upload CSV, PDF, and Word files into managed data sources, then analyze them with governed planning, execution, risk control, audit trails, and reports.",
        "rows": "Rows",
        "cols": "Columns",
        "missing": "Missing",
        "numeric_cols": "Numeric Columns",
        "data_preview": "Data Preview",
        "field_info": "Schema",
        "field": "Field",
        "type": "Type",
        "missing_count": "Missing",
        "unique": "Unique",
        "empty_chart": "No chart data.",
        "chart_truncated": "Chart data was truncated to the display limit.",
        "blocked": "High-risk operation blocked. Confirm the risk in the sidebar before execution.",
        "plan_detail": "Plan Details",
        "tools": "Tools",
        "planner": "Planner",
        "none": "None",
        "result_truncated": "Result rows were truncated.",
        "bar": "Bar Chart",
        "line": "Line Chart",
        "scatter": "Scatter Plot",
        "raw_json": "Raw JSON Result",
        "recent_n": "Recent {n}",
        "no_audit": "No audit records",
        "executed": "Executed",
        "cancelled": "Blocked",
        "failed": "Failed",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "login_failed": "Invalid username or password. Default: admin / admin123",
        "login_required": "Please log in first. Default admin: admin / admin123",
        "admin": "Admin",
        "analyst": "Analyst",
        "viewer": "Viewer",
        "no_dataset_for_field": "No data source is selected, so fields cannot be changed.",
        "file_field_forbidden": "Uploaded file sources cannot be edited in place. Sample sources and imported database sources can be edited.",
        "field_rename_hint": "Please specify the rename, e.g. rename price to total_price.",
        "field_synced": "Field synced to database: {old} -> {new}",
        "field_review": "Field change requires admin approval. Request created: {id}",
        "value_change_hint": "Please specify the field and value, e.g. set category to 1 for all rows.",
        "value_synced": "Field values synced to database: {column} set to {value} for {count} rows.",
        "value_review": "Value change requires admin approval. Request created: {id}",
        "pending_value": "Pending Value Changes",
        "value_request_line": "{user} requested: set {column} to {value} ({dataset})",
        "current_user": "Current user: {username} · {role}",
        "logout": "Logout",
        "language": "Language",
        "config": "Settings",
        "provider": "Model Provider",
        "model": "Model Name",
        "api_ok": "{provider} API Key is set. The app will call {model} first and fall back to local rules if the service fails.",
        "api_empty": "No API Key set. Local rule planning is active.",
        "human_confirm": "Confirm high-risk operations",
        "audit": "Audit Log",
        "quality": "Generate Quality Suggestions",
        "data_sources": "Data Source Management",
        "data_sources_caption": "Choose from managed database sources or upload CSV, PDF, and Word documents. Text PDFs are parsed directly; scanned PDFs are marked as OCR-required.",
        "source_select": "Choose Source",
        "source_library": "Data Source Library",
        "source_upload": "Upload File (CSV / PDF / Word)",
        "source_db": "Import Database Table",
        "no_source": "No data sources yet. Upload a CSV, PDF, or Word file.",
        "choose_source": "Select Data Source",
        "needs_ocr": "This PDF has no readable text layer and may be scanned. It is marked as OCR-required.",
        "doc_preview": "Document Text Preview",
        "db_url": "Database URL",
        "max_rows": "Max Rows to Import",
        "load_tables": "Load Tables",
        "choose_table": "Select Table",
        "import_source": "Import as Data Source",
        "imported_table": "Imported database table: {table}",
        "db_import_failed": "Database import failed: {error}",
        "upload_file": "Upload data file (.csv / .pdf / .docx / .doc)",
        "upload_help": "CSV becomes a structured table. PDF/DOCX is converted to a field/value/confidence table.",
        "saved_source": "Saved as database data source: {name}",
        "upload_ocr": "Uploaded, but the PDF may be scanned and no text layer was found. OCR is required for field extraction.",
        "upload_failed": "Upload parsing failed: {error}",
        "field_governance": "Field Governance",
        "old_field": "Old Field",
        "new_field": "New Field Name",
        "submit_field": "Submit Field Change",
        "pending_field": "Pending Field Changes",
        "request_line": "{user} requested: {old} -> {new} ({dataset})",
        "approve": "Approve",
        "reject": "Reject",
        "approved": "Approved and synced to database.",
        "rejected": "Rejected.",
        "tasks": "Tasks",
        "task": "Task",
        "status": "Status",
        "progress": "Progress",
        "question": "Question",
        "no_tasks": "No tasks",
        "reports": "Report Center",
        "download_report_task": "Download Task Report {id}",
        "no_reports": "No downloadable reports.",
        "users": "Users & Roles",
        "new_username": "New Username",
        "new_password": "New Password",
        "role": "Role",
        "create_user": "Create User",
        "user_created": "User created.",
        "create_failed": "Create failed: {error}",
        "query": "Ask",
        "examples": ["Show data quality profile", "Top 10 rows by price", "Bar chart for furnishingstatus distribution", "Scatter plot of area and price", "Calculate correlations for numeric columns"],
        "input_query": "Enter your analysis question",
        "example_prefix": "Example: ",
        "quick_example": "Quick Example",
        "run": "Run Analysis",
        "choose_first": "Select a database data source or upload a CSV / PDF / Word file first.",
        "query_empty": "Please enter a question.",
        "analyzing": "Planning and executing analysis...",
        "analysis_failed": "Analysis failed: {error}",
        "result": "Analysis Result",
        "task_id": "Task ID: `{id}`",
        "download_html": "Download HTML Report",
    },
}


def current_lang() -> str:
    return "en" if st.query_params.get("lang", "zh") == "en" else "zh"


def tr(key: str, **kwargs: Any) -> str:
    value = I18N[current_lang()].get(key, I18N["zh"].get(key, key))
    if isinstance(value, list):
        return value  # type: ignore[return-value]
    return value.format(**kwargs) if kwargs else value


def get_facade(provider: str, api_key: str, model_name: str) -> DataframeAgentFacade:
    api_key = api_key.strip()
    cached = st.session_state.get("_llm_config", {})
    current = {"provider": provider, "api_key": api_key, "model_name": model_name}
    if "facade" not in st.session_state or cached != current:
        st.session_state["facade"] = DataframeAgentFacade(
            api_key=api_key,
            provider=provider,
            model_name=model_name,
            audit_log_path=APP_DIR / "storage" / "shuxi.db",
        )
        st.session_state["_llm_config"] = current
    return st.session_state["facade"]


def load_csv(file_obj: Any) -> pd.DataFrame:
    return pd.read_csv(file_obj)


def render_dataset(df: pd.DataFrame) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric(tr("rows"), f"{df.shape[0]:,}")
    metric_cols[1].metric(tr("cols"), f"{df.shape[1]:,}")
    metric_cols[2].metric(tr("missing"), f"{int(df.isna().sum().sum()):,}")
    metric_cols[3].metric(tr("numeric_cols"), f"{len(df.select_dtypes(include=['number']).columns):,}")

    with st.expander(tr("data_preview"), expanded=True):
        st.dataframe(df.head(200), use_container_width=True, height=320)

    with st.expander(tr("field_info")):
        info = pd.DataFrame(
            {
                tr("field"): df.columns.astype(str),
                tr("type"): [str(dtype) for dtype in df.dtypes],
                tr("missing_count"): df.isna().sum().astype(int).values,
                tr("unique"): [int(df[col].nunique(dropna=True)) for col in df.columns],
            }
        )
        st.dataframe(info, use_container_width=True, hide_index=True)


def render_chart(payload: Dict[str, Any], chart_type: str) -> None:
    columns = payload.get("columns", [])
    data = payload.get("data", [])
    if not columns or not data:
        st.info(tr("empty_chart"))
        return

    chart_df = pd.DataFrame(data, columns=columns)
    x_col = payload.get("x") or columns[0]
    y_col = payload.get("y") or (columns[1] if len(columns) > 1 else columns[0])
    if x_col not in chart_df.columns or y_col not in chart_df.columns:
        st.dataframe(chart_df, use_container_width=True)
        return

    if payload.get("truncated"):
        st.caption(tr("chart_truncated"))

    if chart_type == "bar":
        fig = px.bar(chart_df, x=x_col, y=y_col)
    elif chart_type == "line":
        fig = px.line(chart_df, x=x_col, y=y_col, markers=True)
    else:
        fig = px.scatter(chart_df, x=x_col, y=y_col, trendline="ols" if len(chart_df) > 2 else None)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: Dict[str, Any]) -> None:
    if result.get("blocked"):
        st.error(tr("blocked"))
        st.json({"risk_factors": result.get("risk_factors", []), "trace_id": result.get("_trace_id")})
        with st.expander(tr("plan_detail")):
            st.json(result.get("plan", {}))
        return

    st.caption(
        "Trace ID: `{}` | {}: {} | {}: {}".format(
            result.get("_trace_id", "N/A"),
            tr("tools"),
            ", ".join(result.get("_executed_tools", [])) or tr("none"),
            tr("planner"),
            result.get("_planner", "unknown"),
        )
    )
    for warning in result.get("_warnings", []):
        st.warning(warning)

    has_structured = any(key in result for key in ("table", "bar", "line", "scatter"))
    if result.get("answer") and not has_structured:
        st.success(str(result["answer"]))

    if "table" in result:
        table = result["table"]
        df_table = pd.DataFrame(table.get("data", []), columns=table.get("columns", []))
        if table.get("truncated"):
            st.caption(tr("result_truncated"))
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    chart_titles = {"bar": tr("bar"), "line": tr("line"), "scatter": tr("scatter")}
    for chart_type, title in chart_titles.items():
        if chart_type in result:
            st.subheader(title)
            render_chart(result[chart_type], chart_type)

    with st.expander(tr("raw_json")):
        st.json(result)


def render_audit_sidebar(facade: DataframeAgentFacade) -> None:
    events = facade.audit.recent(10)
    st.caption(tr("recent_n", n=len(events)))
    if not events:
        st.caption(tr("no_audit"))
        return
    for event in reversed(events):
        status = event.get("status", "")
        label = tr(status) if status in {"executed", "cancelled", "failed"} else status
        st.markdown(f"`{event.get('trace_id')}` {label} · {event.get('risk_level')} · {event.get('intent', '')[:24]}")


def require_login() -> Dict[str, Any]:
    if "user" in st.session_state:
        return st.session_state["user"]
    token = st.query_params.get("token", "")
    user_from_token = auth.user_from_token(token)
    if user_from_token:
        st.session_state["user"] = user_from_token
        st.session_state["auth_token"] = token
        return user_from_token
    with st.sidebar:
        st.header(tr("login"))
        username = st.text_input(tr("username"), value="admin")
        password = st.text_input(tr("password"), type="password")
        if st.button(tr("login"), use_container_width=True):
            user = auth.login(username, password)
            if user:
                token = auth.create_session(user["username"])
                st.session_state["user"] = user
                st.session_state["auth_token"] = token
                st.query_params["token"] = token
                st.rerun()
            st.error(tr("login_failed"))
    st.info(tr("login_required"))
    st.stop()


def role_label(role: str) -> str:
    return tr(role) if role in {"admin", "analyst", "viewer"} else role


def parse_field_rename_intent(query: str, columns: list[str]) -> tuple[str, str] | None:
    if any(token in query for token in ("全部", "所有", "所有值", "替换为")):
        return None
    patterns = [
        r"把\s*([A-Za-z0-9_\u4e00-\u9fa5]+)\s*(?:字段|列名|列)?\s*(?:改成|改为|重命名为)\s*([A-Za-z0-9_\u4e00-\u9fa5]+)",
        r"(?:rename|change)\s+([A-Za-z0-9_]+)\s+(?:to|as)\s+([A-Za-z0-9_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
    for old_name in columns:
        if old_name in query and any(word in query for word in ("改字段", "改列名", "重命名", "rename")):
            return old_name, ""
    return None


def parse_column_value_change_intent(query: str, columns: list[str]) -> tuple[str, Any] | None:
    value_pattern = r"(.+?)(?:。|\.|$)"
    patterns = [
        rf"(?:把|将)?\s*([A-Za-z0-9_\u4e00-\u9fa5]+)\s*(?:字段|列)?\s*(?:的)?\s*(?:所有值|全部值|全部|所有)?\s*(?:改成|改为|替换为|设为|设置为)\s*{value_pattern}",
        rf"(?:set|replace|change)\s+([A-Za-z0-9_]+)\s+(?:to|as|with)\s*{value_pattern}",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        column = match.group(1).strip()
        if column not in columns:
            continue
        return column, _parse_literal_value(match.group(2).strip())
    for column in sorted(columns, key=len, reverse=True):
        if str(column) not in query:
            continue
        for token in ("替换为", "改成", "改为", "设置为", "设为"):
            if token not in query:
                continue
            value = query.rsplit(token, 1)[-1]
            if value.strip():
                return str(column), _parse_literal_value(value)
    return None


def _parse_literal_value(raw: str) -> Any:
    value = raw.strip().strip("，,。.;；")
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def handle_field_change(
    dataset_id: str | None,
    old_name: str,
    new_name: str,
    user: Dict[str, Any],
) -> bool:
    if not dataset_id:
        st.error(tr("no_dataset_for_field"))
        return True
    dataset = storage.get_dataset(dataset_id) or {}
    if dataset.get("source_type") == "upload":
        st.error(tr("file_field_forbidden"))
        return True
    if not old_name or not new_name:
        st.warning(tr("field_rename_hint"))
        return True
    if user["role"] != "admin":
        st.error("字段修改属于高风险操作，仅管理员可执行。当前用户角色为 " + role_label(user["role"]) + "，请联系管理员。")
        return True
    storage.rename_dataset_column(dataset_id, old_name, new_name)
    st.success(tr("field_synced", old=old_name, new=new_name))
    st.session_state["df"] = storage.read_dataset(dataset_id)
    st.rerun()


def handle_value_change(
    dataset_id: str | None,
    column: str,
    value: Any,
    user: Dict[str, Any],
) -> bool:
    if not dataset_id:
        st.error(tr("no_dataset_for_field"))
        return True
    dataset = storage.get_dataset(dataset_id) or {}
    if dataset.get("source_type") == "upload":
        st.error(tr("file_field_forbidden"))
        return True
    if not column:
        st.warning(tr("value_change_hint"))
        return True
    if user["role"] != "admin":
        st.error("数据值修改属于高风险操作，仅管理员可执行。当前用户角色为 " + role_label(user["role"]) + "，请联系管理员。")
        return True
    count = storage.update_dataset_column_values(dataset_id, column, value)
    st.session_state["df"] = storage.read_dataset(dataset_id)
    st.session_state["flash_message"] = tr("value_synced", column=column, value=value, count=count)
    st.rerun()
    return True


def main() -> None:
    st.title(tr("app_title"))
    st.caption(tr("app_caption"))
    if st.session_state.get("flash_message"):
        st.success(st.session_state.pop("flash_message"))
    user = require_login()

    with st.sidebar:
        col_user, col_logout = st.columns([3, 1])
        with col_user:
            st.caption(tr("current_user", username=user["username"], role=role_label(user["role"])))
        with col_logout:
            if st.button(tr("logout"), use_container_width=True):
                auth.logout(st.session_state.get("auth_token", ""))
                st.session_state.pop("user", None)
                st.session_state.pop("auth_token", None)
                st.query_params.clear()
                st.rerun()

        lang_default = st.query_params.get("lang", "zh")
        language = st.selectbox(tr("language"), ["中文", "English"], index=1 if lang_default == "en" else 0)
        selected_lang = "en" if language == "English" else "zh"
        if selected_lang != lang_default:
            st.query_params["lang"] = selected_lang
            st.rerun()

        provider_label = st.selectbox(tr("provider"), ["DeepSeek", "DashScope"], index=0)
        provider = "deepseek" if provider_label == "DeepSeek" else "dashscope"
        default_key = os.getenv("DEEPSEEK_API_KEY", "") if provider == "deepseek" else os.getenv("DASHSCOPE_API_KEY", "")
        default_model = "deepseek-v4-flash" if provider == "deepseek" else "qwen-turbo"
        model_name = st.text_input(tr("model"), value=default_model)
        key_label = "DeepSeek API Key" if provider == "deepseek" else "DashScope API Key"
        api_key = st.text_input(key_label, value=default_key, type="password").strip()
        if api_key:
            st.caption("✅ " + tr("api_ok", provider=provider_label, model=model_name))
        else:
            st.caption("ℹ️ " + tr("api_empty"))

        human_confirmed = st.checkbox(tr("human_confirm"), value=True, help="高风险操作需要人工确认才能执行。取消勾选后，涉及敏感字段、高危工具、大批量导出的分析将被阻断。")

        with st.expander("🔍 " + tr("audit"), expanded=False):
            facade_for_audit = get_facade(provider, api_key, model_name)
            render_audit_sidebar(facade_for_audit)
        with st.expander("💡 " + tr("quality"), expanded=False):
            for tip in facade_for_audit.quality_suggestions():
                st.info(tip)

    st.subheader(tr("data_sources"))
    st.caption(tr("data_sources_caption"))
    source_options = [tr("source_library"), tr("source_upload")]
    if user["role"] == "admin":
        source_options.append(tr("source_db"))
    source_mode = st.radio(tr("source_select"), source_options, horizontal=True)
    df = None
    selected_dataset_id = None
    if source_mode == tr("source_library"):
        datasets = storage.list_datasets()
        if not datasets:
            st.warning(tr("no_source"))
        else:
            options = {
                (
                    f"{item['name']} · {item.get('content_type', 'table')} · "
                    f"{item['rows']} x {item['columns']} · {item.get('extraction_status', 'parsed')}"
                ): item["id"]
                for item in datasets
            }
            selected_label = st.selectbox(tr("choose_source"), list(options.keys()))
            selected_dataset_id = options[selected_label]
            selected_meta = storage.get_dataset(selected_dataset_id) or {}
            df = storage.read_dataset(selected_dataset_id)
            if selected_meta.get("extraction_status") == "needs_ocr":
                st.warning(tr("needs_ocr"))
            if selected_meta.get("text_preview"):
                with st.expander(tr("doc_preview")):
                    st.text(selected_meta["text_preview"][:3000])
    else:
        if source_mode == tr("source_db"):
            try:
                require_role(user, {"admin"})
                db_url = st.text_input(
                    tr("db_url"),
                    placeholder="postgresql+psycopg://user:pass@host:5432/db | mysql+pymysql://user:pass@host/db",
                    type="password",
                )
                limit = st.number_input(tr("max_rows"), min_value=100, max_value=100000, value=10000, step=100)
                if db_url and st.button(tr("load_tables"), use_container_width=True):
                    st.session_state["db_tables"] = list_sql_tables(db_url)
                    st.session_state["db_url"] = db_url
                tables = st.session_state.get("db_tables", [])
                if tables:
                    table_name = st.selectbox(tr("choose_table"), tables)
                    if st.button(tr("import_source"), type="primary", use_container_width=True):
                        selected_dataset_id = storage.import_database_table(st.session_state["db_url"], table_name, int(limit))
                        df = storage.read_dataset(selected_dataset_id)
                        st.success(tr("imported_table", table=table_name))
            except Exception as exc:
                st.error(tr("db_import_failed", error=exc))
        else:
            try:
                require_role(user, {"admin", "analyst"})
            except Exception as exc:
                st.error(str(exc))
                uploaded = None
            else:
                uploaded = st.file_uploader(
                    tr("upload_file"),
                    type=["csv", "pdf", "docx", "doc"],
                    help=tr("upload_help"),
                )
            if uploaded is not None:
                try:
                    content = uploaded.getvalue()
                    selected_dataset_id = storage.save_uploaded_dataset(uploaded.name, content)
                    meta = storage.get_dataset(selected_dataset_id) or {}
                    df = storage.read_dataset(selected_dataset_id)
                    st.success(tr("saved_source", name=uploaded.name))
                    if meta.get("extraction_status") == "needs_ocr":
                        st.warning(tr("upload_ocr"))
                    if meta.get("text_preview"):
                        with st.expander(tr("doc_preview"), expanded=True):
                            st.text(meta["text_preview"][:3000])
                except Exception as exc:
                    st.error(tr("upload_failed", error=exc))
    if df is not None:
        st.session_state["df"] = df
        st.session_state["dataset_id"] = selected_dataset_id
        render_dataset(df)

        if user["role"] == "admin":
            with st.expander(tr("field_governance"), expanded=False):
                selected_meta = storage.get_dataset(selected_dataset_id) if selected_dataset_id else None
                if selected_meta and selected_meta.get("source_type") != "upload":
                    c1, c2 = st.columns(2)
                    old_name = c1.selectbox(tr("old_field"), list(df.columns), key="field_old_name")
                    new_name = c2.text_input(tr("new_field"), key="field_new_name")
                    if st.button(tr("submit_field"), use_container_width=True):
                        handle_field_change(selected_dataset_id, old_name, new_name.strip(), user)
                else:
                    st.info(tr("file_field_forbidden"))

    with st.expander(tr("tasks"), expanded=False):
        tasks = storage.list_tasks(limit=10)
        if tasks:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            tr("task"): item["id"],
                            tr("status"): item["status"],
                            tr("progress"): item["progress"],
                            tr("question"): item["query"],
                        }
                        for item in tasks
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(tr("no_tasks"))

    with st.expander(tr("reports"), expanded=False):
        succeeded = [task for task in storage.list_tasks(limit=50) if task["status"] == "succeeded" and task.get("result")]
        if succeeded:
            for task in succeeded[:10]:
                st.download_button(
                    label=tr("download_report_task", id=task["id"]),
                    data=facade_for_audit.export_report(task["query"], task["result"]).encode("utf-8"),
                    file_name=f"report_{task['id']}.html",
                    mime="text/html",
                    use_container_width=True,
                )
        else:
            st.caption(tr("no_reports"))

    if user["role"] == "admin":
        with st.expander(tr("users"), expanded=False):
            new_user = st.text_input(tr("new_username"))
            new_password = st.text_input(tr("new_password"), type="password")
            role_options = {"analyst": tr("analyst"), "viewer": tr("viewer"), "admin": tr("admin")}
            new_role_label = st.selectbox(tr("role"), list(role_options.values()))
            new_role = next(key for key, value in role_options.items() if value == new_role_label)
            if st.button(tr("create_user"), use_container_width=True):
                try:
                    auth.create_user(new_user, new_password, new_role)
                    st.success(tr("user_created"))
                except Exception as exc:
                    st.error(tr("create_failed", error=exc))

    st.subheader(tr("query"))
    examples = I18N[current_lang()]["examples"]
    query = st.text_area(tr("input_query"), placeholder=tr("example_prefix") + examples[0], height=100)
    selected_example = st.selectbox(tr("quick_example"), [""] + examples)
    if selected_example and not query:
        query = selected_example

    run = st.button(tr("run"), type="primary", use_container_width=True)
    if run:
        try:
            require_role(user, {"admin", "analyst"})
        except Exception as exc:
            st.error(str(exc))
            return
        if "df" not in st.session_state:
            st.warning(tr("choose_first"))
            return
        if not query.strip():
            st.warning(tr("query_empty"))
            return

        columns = list(st.session_state["df"].columns)
        value_change_intent = parse_column_value_change_intent(query.strip(), columns)
        if value_change_intent:
            if handle_value_change(st.session_state.get("dataset_id"), value_change_intent[0], value_change_intent[1], user):
                return

        rename_intent = parse_field_rename_intent(query.strip(), columns)
        if rename_intent:
            if handle_field_change(st.session_state.get("dataset_id"), rename_intent[0], rename_intent[1], user):
                return

        facade = get_facade(provider, api_key, model_name)
        task_id = storage.create_task(query.strip(), st.session_state.get("dataset_id"))
        with st.spinner(tr("analyzing")):
            try:
                storage.update_task(task_id, "running", 30)
                result = facade.analyze(st.session_state["df"], query.strip(), human_confirmed=human_confirmed)
                storage.update_task(task_id, "succeeded", 100, result=result)
            except Exception as exc:
                storage.update_task(task_id, "failed", 100, error=str(exc))
                st.error(tr("analysis_failed", error=exc))
                return

        st.subheader(tr("result"))
        st.caption(tr("task_id", id=task_id))
        render_result(result)
        if not result.get("blocked"):
            report = facade.export_report(query.strip(), result)
            st.download_button(
                tr("download_html"),
                data=report.encode("utf-8"),
                file_name=f"shuxi_report_{result.get('_trace_id', 'analysis')}.html",
                mime="text/html",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
