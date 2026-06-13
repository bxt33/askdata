# 「数析」智能数据分析台

「数析」是一个基于 Streamlit、Pandas 和 DeepSeek/DashScope 的智能数据分析台。系统采用“两阶段 LLM + 受控工具执行 + 风险审计”的架构：LLM 先把自然语言解析成结构化计划，后端再逐步调用注册工具完成真实计算，最后生成统一 JSON 结果交给页面渲染。

## 项目结构

```text
「数析」智能数据分析台/
├── main.py                  # Streamlit 界面，依赖 DataframeAgentFacade
├── utils.py                 # 7 大核心能力，按类拆解
│   ├── ToolMeta             # 工具元数据
│   ├── ToolRegistry         # 工具注册中心
│   ├── AnalysisPlan         # 结构化分析计划
│   ├── PlanEngine           # LLM 第一阶段：自然语言 -> 结构化计划
│   ├── RiskGuard            # 风险评估
│   ├── AuditEvent           # 审计事件
│   ├── AuditLogger          # 人机协同审核 + 审计日志
│   ├── DataAnalyzer         # ReAct 执行层 + LLM 第二阶段
│   ├── QualityOptimizer     # 复盘分析 + 优化建议
│   ├── ReportExporter       # HTML 报告生成
│   └── DataframeAgentFacade # 门面类，统一编排
├── requirements.txt         # 项目依赖
├── house_price.csv          # 示例数据：房价数据（545 条）
├── personal_data.csv        # 示例数据：个人消费数据（22 条）
├── .env.example             # 环境变量示例
├── .gitignore               # 忽略运行产物
├── .streamlit/config.toml   # Streamlit 项目配置
├── scripts/run_ai_agent.ps1 # conda ai_agent 启动脚本
├── scripts/run_api.ps1      # FastAPI 后端启动脚本
├── api.py                   # FastAPI 接口服务
├── storage.py               # SQLite 持久化、数据源库、任务状态
├── Dockerfile               # 容器镜像
├── docker-compose.yml       # UI + API 双服务部署
└── tests/test_pipeline.py   # 核心链路测试
```

## 执行流程

```text
用户界面 main.py
↓ facade.analyze(df, query, human_confirmed)
DataframeAgentFacade
├─ PlanEngine.parse()      自然语言 -> AnalysisPlan（LLM 第一阶段）
├─ RiskGuard.assess()      评估风险等级 high / low
├─ AuditLogger.record()    高风险未确认则记录 cancelled 并阻断
├─ DataAnalyzer.execute()  逐步调用 ToolRegistry 工具
│  └─ LLM 第二阶段          生成最终 JSON 结果
├─ AuditLogger.record()    记录 executed / failed
└─ ReportExporter          可选生成 HTML 报告
↓
render_result()            自动分发 answer/table/bar/line/scatter
↓
QualityOptimizer.suggest() 基于审计日志输出优化建议
```

## 工程化能力

- SQLite 持久化：数据源、任务状态、审计事件统一进入 `storage/shuxi.db`。
- PostgreSQL 数据仓库：配置 `WAREHOUSE_DATABASE_URL` 后，上传数据表会写入 PostgreSQL。
- 多文件/多数据源：样例数据和上传数据都会登记到数据源库。
- 文件上传分析：支持 CSV、PDF、DOCX；CSV 入库为结构化表，PDF/DOCX 抽取为 `field/value/confidence` 表。
- 任务队列状态：每次分析创建任务，记录 queued/running/succeeded/failed 和进度。
- Plotly 图表：条形图、折线图、散点图使用交互式图表渲染。
- 数据清洗和自动洞察：支持缺失值清洗、特征摘要、自动洞察。
- 工具参数 schema 校验：每个关键工具在执行前用 Pydantic 校验参数。
- FastAPI 后端：`api.py` 提供数据源上传、任务提交、任务查询接口。
- 后台任务队列：配置 `REDIS_URL` 后使用 RQ worker；否则使用 FastAPI background task。
- 数据库连接器：支持从 PostgreSQL/MySQL 等 SQLAlchemy 连接串导入表。
- 用户登录和角色权限：默认管理员 `admin / admin123`，角色包括 admin、analyst、viewer。
- PDF 表格识别：优先尝试 PyMuPDF 表格抽取；无表格时转字段/段落表。
- OCR 可选增强：扫描 PDF 会标记 `needs_ocr`，安装 Tesseract/EasyOCR 后可继续扩展识别。
- Docker 部署：`Dockerfile` 和 `docker-compose.yml` 支持 UI + API 双服务部署。

## PDF / Word 字段识别策略

- CSV：天然结构化，上传后直接写入 SQLite 数据表。
- DOCX：读取 Word XML 正文，将 `字段: 值`、`字段：值` 抽取为结构化字段；其他段落作为文本行保留。
- PDF 文本版：使用 PyMuPDF 读取文本层，再做字段抽取。
- PDF 扫描件：如果没有文本层，会标记 `needs_ocr`。这类文件需要接入 OCR，例如 Tesseract、EasyOCR 或 PaddleOCR，再进行表格/字段抽取。

## 启动

推荐使用你本机的 `ai_agent` conda 环境：

```powershell
conda run -n ai_agent python -m streamlit run main.py --server.port 8501 --server.address 127.0.0.1
```

或直接运行脚本：

```powershell
.\scripts\run_ai_agent.ps1
```

打开：

```text
http://127.0.0.1:8501
```

启动 API：

```powershell
.\scripts\run_api.ps1
```

API 地址：

```text
http://127.0.0.1:8001/docs
```

Docker 部署：

```powershell
docker compose up --build
```

## API Key

不填写 API Key 时，系统会使用本地规则规划，仍可完成常见分析。

默认推荐 DeepSeek：

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

填写 DeepSeek 或 DashScope API Key 后：

- `PlanEngine.parse()` 优先使用所选模型进行第一阶段规划；
- `DataAnalyzer.execute()` 在工具计算后进入 LLM 第二阶段生成最终 JSON。
