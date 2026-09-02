"""
Regression tests for analytical filter and aggregation execution.
Verifies that queries with filters/aggregations do NOT fall back to dataset preview ('First 10 Rows').
"""
import os
import pytest
import pandas as pd

from agent.filter_engine import FilterEngine
from agent.nlp_parser import NLPCommandParser
from agent.command_parser import CommandParser
from agent.conversational_analyst import ConversationalAnalystAgent
from agent.command_orchestrator import AutonomousCommandOrchestrator
from backend.app.chat.agent import ChatAgent
from backend.app.chat.validator import ResultValidator


@pytest.fixture
def sales_df():
    """Load basic_sales_test.csv dataset fixture."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "uploads", "basic_sales_test.csv")
    if not os.path.exists(csv_path):
        # Fallback inline creation for standalone test runners
        return pd.DataFrame([
            {"date": "2025-01-01", "product": "Laptop", "region": "North", "sales": 12000, "units": 10, "discount": 5},
            {"date": "2025-01-02", "product": "Phone", "region": "South", "sales": 8000, "units": 20, "discount": 10},
            {"date": "2025-01-03", "product": "Laptop", "region": "North", "sales": 15000, "units": 12, "discount": 5},
            {"date": "2025-01-04", "product": "Tablet", "region": "East", "sales": 6000, "units": 15, "discount": 8},
            {"date": "2025-01-05", "product": "Phone", "region": "South", "sales": 9000, "units": 22, "discount": 10},
            {"date": "2025-01-06", "product": "Laptop", "region": "West", "sales": 18000, "units": 14, "discount": 3},
            {"date": "2025-01-07", "product": "Tablet", "region": "East", "sales": 7000, "units": 17, "discount": 8},
            {"date": "2025-01-08", "product": "Phone", "region": "North", "sales": 11000, "units": 25, "discount": 5},
            {"date": "2025-01-09", "product": "Laptop", "region": "West", "sales": 20000, "units": 16, "discount": 3},
            {"date": "2025-01-10", "product": "Tablet", "region": "South", "sales": 7500, "units": 18, "discount": 7},
        ])
    return pd.read_csv(csv_path)


# ==============================================================================
# Step 10: Specific Regression Tests (Tests 1 to 6)
# ==============================================================================

def test_1_filter_and_aggregation_execution(sales_df):
    """
    Test 1:
    Query: 'Show me only the records where discount is 5 or less. Calculate the total sales and total units for these filtered records.'
    Assertions:
    - filter applied correctly
    - matching rows = 5
    - total sales = 76000
    - total units = 77
    - response does NOT return only first 10 rows
    """
    query = "Show me only the records where discount is 5 or less. Calculate the total sales and total units for these filtered records."
    
    # 1. FilterEngine direct
    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5, f"Expected 5 matching rows, got {res.matching_rows}"
    assert res.aggregations["sales"]["value"] == 76000.0, f"Expected 76,000 sales, got {res.aggregations['sales']['value']}"
    assert res.aggregations["units"]["value"] == 77.0, f"Expected 77 units, got {res.aggregations['units']['value']}"

    # 2. CommandParser
    cp = CommandParser(sales_df)
    cp_res = cp.parse(query)
    assert cp_res.get("type") == "filter_result", f"Expected type 'filter_result', got {cp_res.get('type')}"
    assert cp_res.get("type") != "head", "Response must NOT fall back to 'head' preview"
    assert cp_res.get("matching_rows") == 5
    assert cp_res.get("aggregations", {}).get("sales", {}).get("value") == 76000.0
    assert cp_res.get("aggregations", {}).get("units", {}).get("value") == 77.0

    # 3. ConversationalAnalystAgent
    conv = ConversationalAnalystAgent()
    resp, ev, meta = conv.chat(query, df=sales_df)
    assert meta.get("result", {}).get("matching_rows") == 5
    assert meta.get("result", {}).get("aggregations", {}).get("sales", {}).get("value") == 76000.0
    assert meta.get("result", {}).get("aggregations", {}).get("units", {}).get("value") == 77.0
    assert "76,000" in resp
    assert "77" in resp
    assert "First 10 Rows" not in resp


def test_2_analyze_records_filter_only(sales_df):
    """
    Test 2:
    Query: 'Analyze records where discount is 5 or less.'
    Assertions:
    - filtered dataset contains 5 rows
    """
    query = "Analyze records where discount is 5 or less."
    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5
    assert len(res.filtered_df) == 5
    assert all(res.filtered_df["discount"] <= 5)


def test_3_calculate_total_sales_with_filter(sales_df):
    """
    Test 3:
    Query: 'Calculate total sales where discount is 5 or less.'
    Assertion:
    - total sales = 76000
    """
    query = "Calculate total sales where discount is 5 or less."
    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5
    assert "sales" in res.aggregations
    assert res.aggregations["sales"]["value"] == 76000.0


def test_4_calculate_total_units_with_filter(sales_df):
    """
    Test 4:
    Query: 'Calculate total units where discount is 5 or less.'
    Assertion:
    - total units = 77
    """
    query = "Calculate total units where discount is 5 or less."
    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5
    assert "units" in res.aggregations
    assert res.aggregations["units"]["value"] == 77.0


def test_5_compound_filter_support(sales_df):
    """
    Test 5:
    Query: 'Analyze records where discount is 5 or less and region is North or West.'
    Assertion:
    - compound condition evaluated correctly
    - matching rows = 5
    - total sales = 76000
    - total units = 77
    """
    query = "Analyze records where discount is 5 or less and region is North or West."
    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5
    assert all(res.filtered_df["discount"] <= 5)
    assert all(res.filtered_df["region"].isin(["North", "West"]))
    assert res.filtered_df["sales"].sum() == 76000
    assert res.filtered_df["units"].sum() == 77


def test_6_preview_functionality_preserved(sales_df):
    """
    Test 6:
    Query: 'Show me the first 10 rows'
    Assertion:
    - returns dataset preview correctly
    - does NOT break preview functionality
    """
    query = "Show me the first 10 rows"
    
    # 1. Intent check
    assert not FilterEngine.has_filter_intent(query)
    assert FilterEngine.is_legitimate_preview_request(query)

    # 2. NLPCommandParser check
    nlp = NLPCommandParser()
    intent = nlp.parse(query)
    assert intent.action == "head"

    # 3. CommandParser execution check
    cp = CommandParser(sales_df)
    res = cp.parse(query)
    assert res.get("type") == "head"
    assert "reports" in res


# ==============================================================================
# Step 11: Intent Routing Tests
# ==============================================================================

def test_intent_routing():
    """
    Step 11 Routing assertions:
    - 'Show me the first 10 rows' -> PREVIEW ('head')
    - 'Calculate total sales where discount is 5 or less' -> FILTER + AGGREGATION
    - 'Show me only records where discount is 5 or less' -> FILTER
    """
    nlp = NLPCommandParser()

    # Route 1: Preview
    i1 = nlp.parse("Show me the first 10 rows")
    assert i1.action == "head"

    # Route 2: Filter + Aggregation
    i2 = nlp.parse("Calculate total sales where discount is 5 or less")
    assert i2.action == "filter"
    assert any("discount" in f for f in i2.filters)

    # Route 3: Filter only
    i3 = nlp.parse("Show me only records where discount is 5 or less")
    assert i3.action == "filter"
    assert any("discount" in f for f in i3.filters)


# ==============================================================================
# Natural Language Comparison Syntax Variations (Step 3 & 4)
# ==============================================================================

@pytest.mark.parametrize("query_text", [
    "records where discount <= 5",
    "records where discount is <= 5",
    "records where discount is less than or equal to 5",
    "records where discount at most 5",
    "records where discount 5 or less",
    "records where discount <= 5%",
])
def test_filter_syntax_variations(sales_df, query_text):
    """Verify that all equivalent natural language filter expressions evaluate to matching rows = 5."""
    res = FilterEngine.execute(sales_df, query_text)
    assert res.matching_rows == 5


# ==============================================================================
# Step 8: Result Validation Tests
# ==============================================================================

def test_result_validator_catches_incomplete_preview():
    """Verify ResultValidator flags head responses for analytical queries as INCOMPLETE."""
    validator = ResultValidator()
    
    query = "Show me only the records where discount is 5 or less. Calculate total sales."
    
    # Incomplete response (fallback to head)
    head_resp = {"type": "head", "reports": []}
    is_valid, msg = validator.validate_analytical_execution(query, head_resp)
    assert not is_valid
    assert "INCOMPLETE" in msg

    # Successful response
    success_resp = {"type": "filter_result", "matching_rows": 5, "filter": "discount <= 5"}
    is_valid, msg = validator.validate_analytical_execution(query, success_resp)
    assert is_valid
    assert msg == "SUCCESS"
