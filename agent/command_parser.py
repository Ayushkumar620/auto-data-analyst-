"""
Command Parser - Dispatches user commands (both simple keywords and natural
language) to the appropriate agent module.
"""
from .analyzer import DataAnalyzer
from .visualizer import DataVisualizer
from .predictor import DataPredictor
from .bank_parser import BankStatementParser, is_bank_statement
from .nlp_parser import NLPCommandParser
from .insights import InsightsEngine
from .cleaner import DataCleaner
from .report_generator import ReportGenerator


class CommandParser:
    """Parses user commands and dispatches to the appropriate agent module."""

    # Map command keywords to methods
    COMMANDS = {
        "summary": "summary",
        "summarize": "summary",
        "overview": "summary",
        "stats": "stats",
        "describe": "describe",
        "info": "summary",
        "nulls": "nulls",
        "missing": "nulls",
        "na": "nulls",
        "correlation": "correlation",
        "corr": "correlation",
        "head": "head",
        "first": "head",
        "view": "head",
        "unique": "unique",
        "uniques": "unique",
        "predict": "predict",
        "prediction": "predict",
        "forecast": "forecast",
        "forecasting": "forecast",
        "predict future": "forecast",
        "chart": "chart",
        "plot": "chart",
        "graph": "chart",
        "visualize": "chart",
        "histogram": "histogram",
        "hist": "histogram",
        "scatter": "scatter",
        "bar": "bar",
        "line": "line",
        "trend": "line",
        "pie": "pie",
        "box": "box",
        "boxplot": "box",
        "columns": "columns",
        "words": "text",
        "text": "text",
        "clean": "clean",
        "cleanse": "clean",
        "data cleaning": "clean",
        "insights": "insights",
        "insight": "insights",
        "smart insights": "insights",
        "anomalies": "anomalies",
        "anomaly": "anomalies",
        "outliers": "anomalies",
        "outlier": "anomalies",
        "report": "report",
        "executive report": "report",
        "summary report": "report",
        "download report": "report",
        "pdf": "report",
        "help": "help",
    }

    def __init__(self, data):
        self.data = data
        self.analyzer = DataAnalyzer(data)
        self.visualizer = DataVisualizer(data)
        self.predictor = DataPredictor(data)
        self.insights = InsightsEngine(data)
        self.cleaner = DataCleaner(data)
        self.report_gen = ReportGenerator()
        self.nlp = NLPCommandParser()
        self.parsed_transactions = None

    def parse(self, command):
        """Parse a command string and return the result."""
        command = (command or "").strip()
        if not command:
            return {"type": "error", "message": "No command provided. Type 'help' for options."}

        # Auto-detect bank statements and parse them into structured transactions
        self._maybe_parse_bank_statement()

        # Try simple keyword command first
        simple = self._parse_simple(command)
        if simple is not None:
            return simple

        # Fall back to natural language understanding
        return self._parse_natural_language(command)

    def _maybe_parse_bank_statement(self):
        """If the loaded data is a bank statement, parse it into transactions."""
        try:
            if is_bank_statement(self.data):
                parser = BankStatementParser(self.data)
                transactions = parser.parse()
                if not transactions.empty:
                    self.parsed_transactions = transactions
                    return True
        except Exception:
            pass
        self.parsed_transactions = None
        return False

    def _parse_simple(self, command):
        """Try to parse as a simple keyword command. Returns None if not a keyword."""
        lower = command.lower()
        tokens = lower.split()
        cmd_word = tokens[0]

        # Extract params
        kwargs = {}
        x, y, target = None, None, None
        for p in tokens[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                kwargs[k.strip()] = v.strip()
        x = kwargs.get("x")
        y = kwargs.get("y")
        target = kwargs.get("target")
        periods = kwargs.get("periods")

        # Positional arguments: e.g. "predict continuous_target" or "forecast sensor_output for 4 periods"
        if not target and len(tokens) > 1 and "=" not in tokens[1]:
            main_df = self.predictor._get_main_df() if hasattr(self, "predictor") else None
            if main_df is not None:
                col_match = next((c for c in main_df.columns if c.lower() == tokens[1].lower()), tokens[1])
                target = col_match
            else:
                target = tokens[1]

        for idx, tok in enumerate(tokens[1:], start=1):
            if tok in ("for", "periods", "horizon") and idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                periods = int(tokens[idx + 1])
            elif tok.isdigit() and not periods:
                periods = int(tok)

        action = self.COMMANDS.get(cmd_word)
        if action is None:
            # Try substring match for keyword commands
            for key, act in self.COMMANDS.items():
                if lower == key or lower.startswith(key + " "):
                    action = act
                    break

        if action is None:
            return None

        try:
            if action == "help":
                return self._help()
            elif action == "summary":
                return {"type": "summary", "reports": self._summary_with_transactions()}
            elif action == "describe":
                return {"type": "describe", "reports": self.analyzer.describe()}
            elif action == "stats":
                return {"type": "describe", "reports": self.analyzer.describe()}
            elif action == "nulls":
                return {"type": "nulls", "reports": self.analyzer.nulls()}
            elif action == "correlation":
                return {"type": "correlation", "reports": self.analyzer.correlation()}
            elif action == "head":
                return {"type": "head", "reports": self.analyzer.head()}
            elif action == "unique":
                return {"type": "unique", "reports": self.analyzer.unique_values()}
            elif action == "predict":
                result = self.predictor.predict(target=target)
                return {"type": "predict", "result": result}
            elif action == "forecast":
                try:
                    n_periods = int(periods) if periods else 5
                except (ValueError, TypeError):
                    n_periods = 5
                result = self.predictor.forecast(target=target, periods=n_periods)
                return {"type": "forecast", "result": result}
            elif action == "clean":
                reports = self.cleaner.clean()
                return {"type": "clean", "reports": reports}
            elif action == "insights":
                result = self.insights.generate_smart_insights()
                return {"type": "insights", "result": result}
            elif action == "anomalies":
                parsed = getattr(self, "parsed_transactions", None)
                if parsed is not None and not parsed.empty:
                    result = self.insights.detect_anomalies_in_transactions()
                else:
                    result = self.insights.detect_anomalies()
                return {"type": "anomalies", "result": result}
            elif action == "report":
                result = self.insights.generate_report()
                pdf_b64 = self.report_gen.pdf_to_base64(result)
                result["pdf_base64"] = pdf_b64
                return {"type": "report", "result": result}
            elif action == "columns":
                return {"type": "columns", "reports": self.analyzer.summary()}
            elif action == "text":
                result = self.insights.text_analysis()
                return {"type": "text", "result": result}
            elif action in ("chart", "histogram", "scatter", "bar", "line", "pie", "box", "heatmap", "area"):
                chart_type = kwargs.get("type") or kwargs.get("chart_type") or action
                charts = self.visualizer.chart(chart_type=chart_type, x=x, y=y)
                return {
                    "type": "chart",
                    "charts": charts,
                    "available_types": list(self.visualizer.SUPPORTED_CHARTS.keys()),
                }
        except Exception as e:
            return {"type": "error", "message": f"Error executing command: {str(e)}"}
        return None

    def _summary_with_transactions(self):
        """Return summary reports, using parsed transactions if available."""
        parsed = getattr(self, "parsed_transactions", None)
        if parsed is not None and not parsed.empty:
            df = parsed
            return [{
                "name": "transactions",
                "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                "columns": list(df.columns),
                "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
                "describe": df.describe(include="all").fillna("").to_dict(),
                "nulls": df.isnull().sum().to_dict(),
                "head": df.head(10).to_dict(orient="records"),
            }]
        return self.analyzer.summary()

    def _parse_natural_language(self, command):
        """Parse a natural language command and execute the appropriate analysis."""
        intent = self.nlp.parse(command)
        parsed = getattr(self, "parsed_transactions", None)

        # If transactions were parsed, prefer transaction analysis for financial queries
        if intent.action == "transaction" or (intent.metric and parsed is not None):
            df = parsed if (parsed is not None and not parsed.empty) else None
            if df is not None:
                result = self.insights.transaction_analysis(intent)
                if isinstance(result, dict) and "error" not in result:
                    return {"type": "insight", "result": result, "intent": self._intent_to_dict(intent)}
            result = self.insights.aggregate(intent)
            if isinstance(result, dict) and "error" not in result:
                return {"type": "insight", "result": result, "intent": self._intent_to_dict(intent)}
            return {"type": "error", "message": result.get("error", "Could not analyze this request.")}

        if intent.action == "text":
            result = self.insights.text_analysis()
            return {"type": "text", "result": result}

        if intent.action == "forecast":
            result = self.predictor.forecast(target=intent.target or None)
            return {"type": "forecast", "result": result}

        if intent.action == "predict":
            result = self.predictor.predict(target=intent.target or None)
            return {"type": "predict", "result": result}

        if intent.action == "chart":
            ctype = intent.chart_type or "auto"
            charts = self.visualizer.chart(chart_type=ctype, x=intent.column or None, y=intent.column or None)
            return {"type": "chart", "charts": charts}

        if intent.action == "summary":
            return {"type": "summary", "reports": self._summary_with_transactions()}

        if intent.action == "correlation":
            return {"type": "correlation", "reports": self.analyzer.correlation()}

        if intent.action == "nulls":
            return {"type": "nulls", "reports": self.analyzer.nulls()}

        if intent.action == "unique":
            return {"type": "unique", "reports": self.analyzer.unique_values()}

        if intent.action == "head":
            return {"type": "head", "reports": self.analyzer.head()}

        # Default: try generic aggregation
        if intent.metric or intent.column or intent.amount_type:
            result = self.insights.aggregate(intent)
            if isinstance(result, dict) and "error" not in result:
                return {"type": "insight", "result": result, "intent": self._intent_to_dict(intent)}

        return {
            "type": "error",
            "message": "I couldn't understand that command. Try 'summary', 'chart', 'correlation', 'predict target=col', 'insights', 'anomalies', 'forecast', 'report', or ask like 'total sales', 'how many words', 'spending last month'.",
        }

    def _intent_to_dict(self, intent):
        return {
            "action": intent.action,
            "metric": intent.metric,
            "amount_type": intent.amount_type,
            "time_filter": intent.time_filter,
            "group_by": intent.group_by,
            "target": intent.target,
            "column": intent.column,
            "chart_type": intent.chart_type,
        }

    def _help(self):
        return {
            "type": "help",
            "commands": [
                {"command": "summary", "description": "Full overview (shape, columns, dtypes, sample, nulls)"},
                {"command": "describe", "description": "Statistical summary of numeric columns"},
                {"command": "nulls", "description": "Missing value analysis"},
                {"command": "correlation", "description": "Correlation matrix of numeric columns"},
                {"command": "head", "description": "Show first 10 rows"},
                {"command": "unique", "description": "Unique value counts per column"},
                {"command": "chart", "description": "Auto-generate a chart (x=col y=col)"},
                {"command": "words", "description": "Text analysis (word count, sentences, frequency)"},
                {"command": "clean", "description": "Auto-clean data (missing values, duplicates, types, outliers)"},
                {"command": "insights", "description": "Generate smart natural-language insights and recommendations"},
                {"command": "anomalies", "description": "Detect statistical anomalies and outliers"},
                {"command": "forecast target=col", "description": "Forecast future values of a numeric column (periods=N)"},
                {"command": "report", "description": "Generate an executive report with downloadable PDF"},
                {"command": "predict target=col", "description": "Train a model to predict a column"},
                {"command": "total sales", "description": "Natural language: sum a column"},
                {"command": "spending last month", "description": "Natural language: analyze transactions over time"},
                {"command": "how many words", "description": "Natural language: text word count"},
                {"command": "chart by category", "description": "Natural language: chart grouped by category"},
            ],
        }
