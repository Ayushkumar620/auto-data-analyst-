"""
Regression tests for analytical filter and aggregation execution.
Verifies that queries with filters/aggregations do NOT fall back to dataset preview ('First 10 Rows'),
and handles compound natural-language OR/EITHER expressions with proper boolean AST and dimensional breakdowns.
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
# Step 11: Simple Filter Preservation
# ==============================================================================

def test_1_filter_and_aggregation_execution(sales_df):
    """
    Test 1 (Step 11):
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


def test_6_preview_functionality_preserved(sales_df):
    """
    Test 6:
    Query: 'Show me the first 10 rows'
    Assertion:
    - returns dataset preview correctly
    - does NOT break preview functionality
    """
    query = "Show me the first 10 rows"
    assert not FilterEngine.has_filter_intent(query)
    assert FilterEngine.is_legitimate_preview_request(query)

    nlp = NLPCommandParser()
    intent = nlp.parse(query)
    assert intent.action == "head"

    cp = CommandParser(sales_df)
    res = cp.parse(query)
    assert res.get("type") == "head"
    assert "reports" in res


# ==============================================================================
# Step 7: "EITHER" is a Logical Word, Never a Value
# ==============================================================================

def test_step7_either_is_logical_word_never_value(sales_df):
    """
    Verify that 'either' is treated as a logical modifier and NEVER parsed as a value.
    'region is either North or West' must NOT produce region == 'either'.
    """
    query = "records where the region is either North or West"
    f = FilterEngine.parse_filters(query, list(sales_df.columns), sales_df)
    assert f is not None

    def check_no_either(node):
        if hasattr(node, "value"):
            val = node.value
            if isinstance(val, (list, tuple)):
                assert "either" not in [v.lower() for v in val], "List value must not contain 'either'"
            else:
                assert str(val).lower() != "either", "Value must not be 'either'"
        if hasattr(node, "conditions"):
            for child in node.conditions:
                check_no_either(child)

    check_no_either(f)

    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5, f"Expected 5 rows, got {res.matching_rows}"
    assert "region == 'either'" not in res.filter_description


# ==============================================================================
# Step 8: Support Multiple OR Values
# ==============================================================================

def test_step8_multiple_or_values(sales_df):
    """
    Test 3 values in categorical OR:
    - 'region is either North, West, or South' -> North (4) + West (2) + South (3) = 9 rows (out of 10, East has 2 rows)
    """
    # Test 'either North, West, or South'
    q1 = "records where the region is either North, West, or South"
    r1 = FilterEngine.execute(sales_df, q1)
    assert r1.matching_rows == 8, f"Expected 8 rows (all except East), got {r1.matching_rows}"
    assert all(r1.filtered_df["region"].isin(["North", "West", "South"]))

    # Test 'region is North or West or South'
    q2 = "records where region is North or West or South"
    r2 = FilterEngine.execute(sales_df, q2)
    assert r2.matching_rows == 8
    assert all(r2.filtered_df["region"].isin(["North", "West", "South"]))


# ==============================================================================
# Step 9: AND / OR Combinations
# ==============================================================================

def test_step9_and_or_combinations(sales_df):
    """
    Test various combinations of AND and OR:
    1. discount <= 5 AND region is North or West
    2. discount <= 5 AND product is Laptop or Phone
    3. region is North or West AND product is Laptop or Phone
    4. discount <= 5 AND (region is North or West)
    """
    # 1. discount <= 5 AND region is North or West -> 5 rows, 76,000 sales, 77 units
    r1 = FilterEngine.execute(sales_df, "records where discount <= 5 and region is North or West. Calculate total sales and total units.")
    assert r1.matching_rows == 5
    assert r1.aggregations["sales"]["value"] == 76000.0
    assert r1.aggregations["units"]["value"] == 77.0

    # 2. discount <= 5 AND product is Laptop or Phone -> 5 rows
    r2 = FilterEngine.execute(sales_df, "records where discount <= 5 and product is Laptop or Phone")
    assert r2.matching_rows == 5
    assert all(r2.filtered_df["product"].isin(["Laptop", "Phone"]))
    assert all(r2.filtered_df["discount"] <= 5)

    # 3. region is North or West AND product is Laptop or Phone -> 5 rows
    r3 = FilterEngine.execute(sales_df, "records where region is North or West and product is Laptop or Phone")
    assert r3.matching_rows == 5
    assert all(r3.filtered_df["region"].isin(["North", "West"]))
    assert all(r3.filtered_df["product"].isin(["Laptop", "Phone"]))

    # 4. discount <= 5 AND (region is North or West) -> 5 rows
    r4 = FilterEngine.execute(sales_df, "records where discount <= 5 and (region is North or West)")
    assert r4.matching_rows == 5


# ==============================================================================
# Step 12: Compound Test (Step 12 exact specification)
# ==============================================================================

def test_step12_compound_test(sales_df):
    """
    Exact Step 12 specification:
    Query:
    'Analyze sales for records where the discount is 5% or less and the region is either North or West.
     Calculate the total sales, total units sold, and number of records matching these conditions.'
    Expected:
    - Filter: discount <= 5 AND (region == North OR region == West)
    - Rows: 5
    - Sales: 76,000
    - Units: 77
    """
    query = (
        "Analyze sales for records where the discount is 5% or less and the region is either North or West. "
        "Calculate the total sales, total units sold, and number of records matching these conditions."
    )
    res = FilterEngine.execute(sales_df, query)

    assert res.matching_rows == 5, f"Expected 5 matching rows, got {res.matching_rows}"
    assert "discount <= 5" in res.filter_description
    assert "North" in res.filter_description and "West" in res.filter_description
    assert "either" not in res.filter_description

    assert res.aggregations["sales"]["value"] == 76000.0, f"Expected 76,000 sales, got {res.aggregations['sales']['value']}"
    assert res.aggregations["units"]["value"] == 77.0, f"Expected 77 units, got {res.aggregations['units']['value']}"


# ==============================================================================
# Step 13: Full Analysis Test (Breakdowns & Regional Averages)
# ==============================================================================

def test_step13_full_analysis_test(sales_df):
    """
    Exact Step 13 specification:
    Complete original query:
    'Analyze sales for records where the discount is 5% or less and the region is either North or West.
     Calculate the total sales, total units sold, and number of records matching these conditions.
     Break down the filtered results by product, showing total sales and total units for each product.
     Also calculate the average sales for each matching region and identify which region has the highest average sales.
     Do not include records that fail either condition, and do not make causal claims.'
    Expected:
    OVERALL:
    - Matching rows = 5
    - Total sales = 76,000
    - Total units = 77
    PRODUCT:
    - Laptop: sales = 65,000, units = 52
    - Phone: sales = 11,000, units = 25
    REGION:
    - North: average sales = 12,666.67
    - West: average sales = 19,000.00
    - Highest average sales region: West
    """
    query = (
        "Analyze sales for records where the discount is 5% or less and the region is either North or West. "
        "Calculate the total sales, total units sold, and number of records matching these conditions. "
        "Break down the filtered results by product, showing total sales and total units for each product. "
        "Also calculate the average sales for each matching region and identify which region has the highest average sales. "
        "Do not include records that fail either condition, and do not make causal claims."
    )
    res = FilterEngine.execute(sales_df, query)

    # 1. Overall metrics
    assert res.matching_rows == 5
    assert res.aggregations["sales"]["value"] == 76000.0
    assert res.aggregations["units"]["value"] == 77.0

    # 2. Product breakdown
    assert "product" in res.breakdowns
    p_recs = {r["product"]: r for r in res.breakdowns["product"]["records"]}
    assert "Laptop" in p_recs
    assert p_recs["Laptop"]["sales"] == 65000.0
    assert p_recs["Laptop"]["units"] == 52.0

    assert "Phone" in p_recs
    assert p_recs["Phone"]["sales"] == 11000.0
    assert p_recs["Phone"]["units"] == 25.0

    # 3. Regional averages
    assert "region" in res.breakdowns
    r_recs = {r["region"]: r for r in res.breakdowns["region"]["records"]}
    assert "North" in r_recs
    assert abs(r_recs["North"]["sales"] - 12666.67) < 0.1
    assert "West" in r_recs
    assert abs(r_recs["West"]["sales"] - 19000.0) < 0.1

    # 4. Highest average region
    assert res.breakdowns["region"]["highest"]["region"] == "West"
    assert res.breakdowns["region"]["highest"]["value"] == 19000.0


# ==============================================================================
# Step 15: Arbitrary Dataset Independence (No Hardcoding)
# ==============================================================================

def test_step15_arbitrary_dataset_independence():
    """
    Verify parser works on arbitrary datasets with completely different columns, types, and values.
    """
    emp_df = pd.DataFrame([
        {"employee": "Alice", "department": "Engineering", "salary": 120000, "active": True},
        {"employee": "Bob", "department": "Marketing", "salary": 80000, "active": True},
        {"employee": "Charlie", "department": "Sales", "salary": 95000, "active": False},
        {"employee": "Dave", "department": "Engineering", "salary": 140000, "active": True},
        {"employee": "Eve", "department": "HR", "salary": 70000, "active": False},
        {"employee": "Frank", "department": "Sales", "salary": 110000, "active": True},
    ])

    query = (
        "Analyze employees where salary >= 90000 and department is either Engineering or Sales. "
        "Calculate the total salary."
    )
    res = FilterEngine.execute(emp_df, query)

    # Alice (120k), Charlie (95k), Dave (140k), Frank (110k) = 4 matching
    # Total salary = 465,000
    assert res.matching_rows == 4
    assert res.aggregations["salary"]["value"] == 465000.0
    assert "department == 'Engineering'" in res.filter_description
    assert "department == 'Sales'" in res.filter_description


# ==============================================================================
# Natural Language Comparison Syntax Variations
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
# Step 8 / 10: Result Validation Tests
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
