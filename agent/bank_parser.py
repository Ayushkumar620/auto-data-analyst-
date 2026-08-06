"""
Bank Statement Parser - Converts unstructured bank/UPI statement text into
structured transaction data with Date, Time, Description, Type, Category, and Amount.
"""
import re
from datetime import datetime
import pandas as pd


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_amount(raw):
    """Parse an amount string like '+ Rs.1,801.07' or 'India - 87+ Rs.1,700'
    -> (type, float)."""
    raw = raw.replace(",", "").strip()
    # Look for the sign immediately before 'Rs.'
    sign_match = re.search(r"([+-])\s*Rs\.", raw)
    if sign_match:
        amount_type = "received" if sign_match.group(1) == "+" else "paid"
    elif raw.startswith("+"):
        amount_type = "received"
    else:
        amount_type = "paid"
    # Extract the numeric part after Rs.
    rs_match = re.search(r"Rs\.\s*([-+]?\d+\.?\d*)", raw)
    if not rs_match:
        return None, None
    val = float(rs_match.group(1))
    if amount_type == "paid":
        val = -abs(val)
    else:
        val = abs(val)
    return amount_type, val


def parse_date(raw, year=None):
    """Parse a date string like '05 Aug' or '05 Aug'26'."""
    raw = raw.strip()
    # Match patterns like '05 Aug' or '05 Aug'26'
    m = re.match(r"(\d{1,2})\s+([A-Za-z]{3,})", raw)
    if not m:
        return None
    day = int(m.group(1))
    month_str = m.group(2)[:3].lower()
    month = MONTHS.get(month_str)
    if not month:
        return None
    # Check for year in 'yy' form
    ym = re.search(r"'(\d{2})$", raw)
    if ym:
        yy = int(ym.group(1))
        y = 2000 + yy
    elif year:
        y = year
    else:
        y = datetime.now().year
    try:
        return datetime(y, month, day)
    except ValueError:
        return None


class BankStatementParser:
    """Parses bank/UPI statement text into structured transactions."""

    # Keywords that indicate a transaction action line
    ACTION_KEYWORDS = [
        "paid to", "received from", "money sent to", "debited",
        "credited", "transfer to", "money received", "refund",
        "cashback", "upi ref no", "paid", "sent to",
    ]

    def __init__(self, df):
        """
        df: DataFrame with at least a 'text' column containing statement lines,
        or a DataFrame with raw text lines.
        """
        self.df = df

    def _get_lines(self):
        """Extract a clean list of text lines."""
        if "text" in self.df.columns:
            return [str(x).strip() for x in self.df["text"].tolist() if str(x).strip()]
        # Fallback: use first string column
        for col in self.df.columns:
            if self.df[col].dtype == object:
                return [str(x).strip() for x in self.df[col].tolist() if str(x).strip()]
        return []

    def parse(self):
        """Parse the statement and return a structured DataFrame of transactions."""
        lines = self._get_lines()
        if not lines:
            return pd.DataFrame()

        # First pass: find the year context from the header (e.g., "6 AUG'25 - 5 AUG'26")
        year = None
        for line in lines[:50]:
            if "'" in line and re.search(r"\d{2}'?\d{2}", line):
                years = re.findall(r"(\d{2})", line)
                if len(years) >= 2:
                    year = 2000 + int(years[0])
                    break

        transactions = []
        current = None
        i = 0
        n = len(lines)

        # Transaction date template: day + month, optionally followed by year
        date_re = re.compile(r"^\d{1,2}\s+[A-Za-z]{3,}")

        while i < n:
            line = lines[i]

            # Detect a new transaction start: a date line
            if date_re.match(line) and current is not None:
                # If we already have an amount, finalize previous transaction
                if current.get("amount") is not None:
                    transactions.append(current)
                else:
                    # Previous transaction incomplete (no amount) - merge
                    current = None
                current = {"date": None, "time": None, "description": "", "type": None, "category": None, "amount": None}
            elif current is None and date_re.match(line):
                current = {"date": None, "time": None, "description": "", "type": None, "category": None, "amount": None}

            if current is None:
                i += 1
                continue

            # Parse date line
            if date_re.match(line):
                d = parse_date(line, year)
                if d:
                    current["date"] = d
                i += 1
                continue

            # Parse time + action on same or next lines
            # Time pattern: '9:48 PM' or '12:56 PM'
            time_match = re.search(r"(\d{1,2}:\d{2})\s*(AM|PM)", line)
            if time_match:
                current["time"] = time_match.group(0)
                rest = line[time_match.end():].strip()
                if rest:
                    current["description"] = rest

            # Detect action line
            lower = line.lower()
            if any(kt in lower for kt in self.ACTION_KEYWORDS):
                if not current["description"]:
                    current["description"] = line
                # Determine type
                if "received" in lower or "credited" in lower or "cashback" in lower or "refund" in lower or "money received" in lower:
                    if current["type"] is None:
                        current["type"] = "received"
                elif "paid" in lower or "debited" in lower or "sent to" in lower or "transfer to" in lower:
                    if current["type"] is None:
                        current["type"] = "paid"

            # Detect amount line: contains 'Rs.' with +/- prefix
            if "rs." in lower:
                atype, amt = parse_amount(line)
                if amt is not None:
                    current["amount"] = amt
                    if current["type"] is None:
                        current["type"] = atype

            # Detect category/tag: '# SomeTag'
            tag_match = re.search(r"#\s*([A-Za-z\s]+)", line)
            if tag_match and current.get("category") is None:
                current["category"] = tag_match.group(1).strip()

            i += 1

        # Finalize last transaction
        if current and current.get("amount") is not None:
            transactions.append(current)

        if not transactions:
            return pd.DataFrame()

        df = pd.DataFrame(transactions)
        # Drop transactions without amount or date
        df = df[df["amount"].notna()]
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["type"] = df["type"].fillna(df["amount"].apply(lambda a: "received" if a > 0 else "paid"))
        df["category"] = df["category"].fillna("Uncategorized")
        df["description"] = df["description"].fillna("")
        return df.reset_index(drop=True)


def is_bank_statement(df):
    """Heuristically detect if a DataFrame looks like a bank statement."""
    if "text" not in df.columns:
        return False
    text = " ".join(str(x) for x in df["text"].head(200).tolist()).lower()
    # Look for statement + transaction indicators
    indicators = [
        "statement", "upi", "paid to", "received from", "balance",
        "transaction", "passbook", "paytm", "bank", "ref no",
    ]
    score = sum(1 for ind in indicators if ind in text)
    return score >= 2
