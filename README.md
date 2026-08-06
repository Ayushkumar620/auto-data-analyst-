# 🤖 Auto Data Analyst Agent

An automated data analysis agent that works on **any type or form of data** and produces outputs based on **simple user commands**. Built with Python and Flask.

## ✨ Features

- **Multiple data formats**: CSV, Excel (`.xlsx`/`.xls`), JSON, Text, SQLite databases, and even **video files** (extracts metadata & brightness analysis)
- **Simple commands**: `summary`, `describe`, `nulls`, `correlation`, `head`, `unique`, `chart`, `predict`, and more
- **Visualizations**: auto-generated charts (bar, line, scatter, histogram, pie, box)
- **Statistical reports**: summaries, missing-value analysis, correlation matrices
- **Predictions**: train regression/classification models with scikit-learn
- **Autonomous intelligence**: auto data cleaning, natural-language insights, anomaly detection, time-series forecasting, and downloadable executive PDF reports
- **Web interface**: drag-and-drop file upload with a dark, modern UI

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### 1. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

## 📖 Available Commands

| Command | Description |
|---------|-------------|
| `summary` | Full overview (shape, columns, types, preview, nulls) |
| `describe` | Statistical summary of numeric columns |
| `nulls` | Missing value analysis |
| `correlation` | Correlation matrix of numeric columns |
| `head` | Show first 10 rows |
| `unique` | Unique value counts per column |
| `chart` | Auto-generate a chart |
| `histogram` | Histogram of first numeric column |
| `scatter x=colx y=coly` | Scatter plot |
| `bar x=col` | Bar chart of column values |
| `line x=colx y=coly` | Line chart / trend |
| `pie x=col` | Pie chart of a column |
| `box` | Box plot of numeric columns |
| `predict target=col` | Train a model to predict a column |
| `clean` | Auto-clean data (missing values, duplicates, types, outliers) |
| `insights` | Generate smart natural-language insights and recommendations |
| `anomalies` | Detect statistical anomalies and outliers |
| `forecast target=col periods=N` | Forecast future values of a numeric column |
| `report` | Generate an executive report with downloadable PDF |
| `help` | Show all commands |

## 🎯 Usage Examples

1. **Upload** any supported file (e.g., `sample_data.csv`)
2. **Type a command** and click Analyze:
   - `summary` → see the full data overview
   - `correlation` → see relationships between numeric columns
   - `chart` → auto-generate a visualization
   - `predict target=sales` → train a model to predict sales

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`PRD.md`](PRD.md) | Product Requirements — problem statement, target users, features, MVP scope |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design — frontend, backend, AI multi-agent workflow, DB schema, APIs, data flow |

> **Vision:** Turn ADAA into a full AI-powered platform — "Upload your data. Ask questions. Get business insights in minutes."

## 📁 Project Structure

```
auto-data-analyst/
├── app.py                  # Flask web server
├── agent/
│   ├── __init__.py
│   ├── loader.py           # Data loading (all formats)
│   ├── analyzer.py         # Statistical analysis
│   ├── visualizer.py       # Chart generation
│   ├── predictor.py        # ML predictions & forecasting
│   ├── cleaner.py          # Auto data cleaning
│   ├── insights.py         # Smart insights, anomalies, reports
│   ├── report_generator.py # PDF report generation (reportlab)
│   ├── agents.py           # Multi-agent system (specialized agents)
│   ├── base.py             # Base agent class
│   ├── nlp_parser.py       # Natural-language intent parsing
│   ├── bank_parser.py      # Bank/UPI statement parsing
│   ├── llm_router.py       # Optional LLM command routing
│   ├── config.py           # API key / LLM configuration
│   └── command_parser.py   # Command handling & dispatch
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── css/style.css       # Styles
├── uploads/                # Uploaded files
├── sample_data.csv         # Sample dataset
├── requirements.txt
└── README.md
```

## 🔧 Troubleshooting

- **Dependencies not installed**: Run `pip install -r requirements.txt`
- **Large video files**: The web app has a 100MB upload limit (configurable in `app.py`)
- **Chart errors**: Ensure your data has numeric columns for numeric charts

## 🛠️ Customization

- Add more commands in `agent/command_parser.py`
- Add more data formats in `agent/loader.py`
- Adjust upload size limit in `app.py` (`MAX_CONTENT_LENGTH`)

---
Built with ❤️ using Python, Flask, pandas, matplotlib & scikit-learn
