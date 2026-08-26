"""
NLP Command Parser - Understands natural language commands and extracts
intent, metrics, time filters, and aggregation targets from them.
"""
import re
from datetime import datetime, timedelta


MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


class CommandIntent:
    """Represents the parsed intent of a natural language command."""

    def __init__(self):
        self.action = ""            # summary, chart, predict, text, transaction, etc.
        self.metric = ""            # total, sum, average, count, max, min
        self.amount_type = ""       # paid, received, debit, credit
        self.time_filter = {}       # dict with start/end or month/year
        self.group_by = ""          # column to group by
        self.target = ""            # target column for prediction
        self.column = ""            # specific column of interest
        self.chart_type = ""        # bar, line, pie, scatter, histogram
        self.limit = None
        self.raw = ""


class NLPCommandParser:
    """Extracts structured intent from natural language commands."""

    # Metric keywords
    METRIC_PATTERNS = {
        "total": ["total", "sum", "overall", "overall sum"],
        "average": ["average", "avg", "mean"],
        "count": ["count", "how many", "number of", "number"],
        "maximum": ["max", "maximum", "highest", "largest", "top"],
        "minimum": ["min", "minimum", "lowest", "smallest", "least"],
    }

    # Amount/transaction type keywords
    TYPE_KEYWORDS = {
        "received": ["received", "credit", "credited", "incoming", "money in", "income", "earned"],
        "paid": ["paid", "debit", "debited", "spent", "spending", "expense", "expenses", "outgoing", "sent"],
    }

    # Chart type keywords
    CHART_KEYWORDS = {
        "bar": ["bar", "bar chart"],
        "line": ["line", "line chart", "trend", "timeline"],
        "pie": ["pie", "pie chart"],
        "scatter": ["scatter", "scatter plot"],
        "histogram": ["histogram", "distribution", "hist"],
    }

    # Action indicators
    FORECAST_KEYWORDS = ["forecast", "future", "next periods", "what will happen", "next month", "next quarter", "projection", "project"]
    PREDICT_KEYWORDS = ["predict", "prediction", "classify", "classification", "machine learning", "ml", "estimate"]
    SUMMARY_KEYWORDS = ["summary", "summarize", "overview", "describe", "info", "statistics", "stats"]
    HEAD_KEYWORDS = ["head", "first", "show", "view", "display", "table", "rows", "sample"]
    UNIQUE_KEYWORDS = ["unique", "distinct", "categories", "category list"]
    NULL_KEYWORDS = ["null", "missing", "na values", "missing values", "empty"]
    CORRELATION_KEYWORDS = ["correlation", "relationship", "corr", "related"]
    CHART_KEYWORDS_LIST = ["chart", "plot", "graph", "visualize", "visualization", "visual"]

    # Time period keywords
    TIME_KEYWORDS = [
        "today", "yesterday", "this week", "last week", "this month", "last month",
        "this year", "last year", "previous month", "previous year", "past month",
        "past week", "past year", "last 7 days", "last 30 days", "last 90 days",
        "last 12 months", "this quarter", "last quarter",
    ]

    def __init__(self):
        self.intent = CommandIntent()

    def parse(self, command):
        """Parse a natural language command into a CommandIntent."""
        self.intent = CommandIntent()
        self.intent.raw = command
        text = command.lower().strip()

        # Detect action
        self._detect_action(text)
        # Detect metric
        self._detect_metric(text)
        # Detect amount type
        self._detect_type(text)
        # Detect time filter
        self._detect_time(text)
        # Detect group by
        self._detect_group_by(text)
        # Detect chart type
        self._detect_chart_type(text)
        # Detect target column for prediction
        self._detect_target(text)
        # Detect column of interest
        self._detect_column(text)

        return self.intent

    def _detect_action(self, text):
        if any(kw in text for kw in self.FORECAST_KEYWORDS):
            self.intent.action = "forecast"
        elif any(kw in text for kw in self.PREDICT_KEYWORDS):
            self.intent.action = "predict"
        elif any(kw in text for kw in self.SUMMARY_KEYWORDS):
            self.intent.action = "summary"
        elif any(kw in text for kw in self.CHART_KEYWORDS_LIST) or any(ck in text for ck in self.CHART_KEYWORDS):
            self.intent.action = "chart"
        elif any(kw in text for kw in self.CORRELATION_KEYWORDS):
            self.intent.action = "correlation"
        elif any(kw in text for kw in self.NULL_KEYWORDS):
            self.intent.action = "nulls"
        elif any(kw in text for kw in self.UNIQUE_KEYWORDS):
            self.intent.action = "unique"
        elif any(kw in text for kw in self.HEAD_KEYWORDS) and any(w in text for w in ["data", "rows", "table", "record"]):
            self.intent.action = "head"
        # If it's a transaction/financial query, default to transaction analysis
        elif any(w in text for w in ["paid", "received", "spent", "expense", "spending", "revenue", "income", "transaction", "upi"]):
            self.intent.action = "transaction"
        elif any(w in text for w in ["word", "words", "text", "sentence", "sentence count", "character", "chars"]):
            self.intent.action = "text"
        elif not self.intent.action:
            # Default to summary if it's a general question
            if any(w in text for w in ["what", "how", "show", "tell", "which", "when", "give"]):
                self.intent.action = "summary"

    def _detect_metric(self, text):
        for metric, keywords in self.METRIC_PATTERNS.items():
            for kw in keywords:
                if kw in text:
                    self.intent.metric = metric
                    return

    def _detect_type(self, text):
        for val_type, keywords in self.TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    self.intent.amount_type = val_type
                    return

    def _detect_time(self, text):
        now = datetime.now()
        # Specific month mention
        for name, month_num in MONTH_NAMES.items():
            if name in text or (len(name) == 3 and name in text):
                # Check for year
                year_match = re.search(r"\b(19|20)(\d{2})\b", text)
                year = int(year_match.group(0)) if year_match else now.year
                self.intent.time_filter = {
                    "period": "month",
                    "month": month_num,
                    "year": year,
                }
                return

        if "this week" in text:
            start = now - timedelta(days=now.weekday())
            self.intent.time_filter = {"period": "week", "start": start, "end": now}
        elif "last week" in text or "previous week" in text:
            start = now - timedelta(days=now.weekday() + 7)
            end = now - timedelta(days=now.weekday() + 1)
            self.intent.time_filter = {"period": "week", "start": start, "end": end}
        elif "this month" in text:
            self.intent.time_filter = {"period": "month", "start": now.replace(day=1), "end": now}
        elif "last month" in text or "previous month" in text:
            first = now.replace(day=1)
            last_month_end = first - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            self.intent.time_filter = {"period": "month", "start": last_month_start, "end": last_month_end}
        elif "this year" in text:
            self.intent.time_filter = {"period": "year", "start": now.replace(month=1, day=1), "end": now}
        elif "last year" in text or "previous year" in text:
            self.intent.time_filter = {"period": "year", "start": now.replace(year=now.year-1, month=1, day=1), "end": now.replace(year=now.year-1, month=12, day=31)}
        elif "today" in text:
            self.intent.time_filter = {"period": "day", "start": now.replace(hour=0, minute=0, second=0), "end": now}
        elif "yesterday" in text:
            y = now - timedelta(days=1)
            self.intent.time_filter = {"period": "day", "start": y.replace(hour=0, minute=0, second=0), "end": y.replace(hour=23, minute=59, second=59)}
        elif "last 7 days" in text or "past week" in text:
            self.intent.time_filter = {"period": "days", "start": now - timedelta(days=7), "end": now}
        elif "last 30 days" in text or "past month" in text:
            self.intent.time_filter = {"period": "days", "start": now - timedelta(days=30), "end": now}
        elif "last 90 days" in text or "past year" in text or "last 12 months" in text:
            self.intent.time_filter = {"period": "days", "start": now - timedelta(days=365), "end": now}

    def _detect_group_by(self, text):
        # "by X" or "per X" or "for each X"
        match = re.search(r"\b(by|per|for each|group by)\s+([a-z_]+)", text)
        if match:
            self.intent.group_by = match.group(2)

    def _detect_chart_type(self, text):
        for chtype, keywords in self.CHART_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    self.intent.chart_type = chtype
                    return

    def _detect_target(self, text):
        # "predict X" or "forecast X"
        match = re.search(r"\b(predict|forecast)\s+([a-z_]+)", text)
        if match:
            self.intent.target = match.group(2)

    def _detect_column(self, text):
        # Look for known data column names mentioned (heuristic)
        # e.g., "sales", "revenue", "price", "amount", "age", "score"
        known = ["sales", "revenue", "price", "amount", "age", "score", "profit",
                 "quantity", "units", "cost", "value", "rating"]
        for col in known:
            if col in text:
                self.intent.column = col
                return
