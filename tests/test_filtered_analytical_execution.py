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


# ==============================================================================
# Date Filtering Regression Tests (Steps 6 to 13)
# ==============================================================================

def test_date_1_single_date(sales_df):
    """
    Step 6: Single Date Test
    Query: 'Show me only the records from January 3, 2025. Calculate the total sales and total units for that date.'
    Expected:
    - filter: date == 2025-01-03
    - rows: 1
    - sales: 15,000
    - units: 12
    - matching row: 2025-01-03 | Laptop | North | 15000 | 12 | discount 5
    """
    query = "Show me only the records from January 3, 2025. Calculate the total sales and total units for that date."
    res = FilterEngine.execute(sales_df, query)

    assert res.matching_rows == 1, f"Expected 1 matching row, got {res.matching_rows}"
    assert "date == 2025-01-03" in res.filter_description
    assert res.aggregations["sales"]["value"] == 15000.0
    assert res.aggregations["units"]["value"] == 12.0
    assert len(res.filtered_df) == 1
    matched_row = res.filtered_df.iloc[0]
    assert matched_row["date"] == "2025-01-03"
    assert matched_row["product"] == "Laptop"
    assert matched_row["sales"] == 15000
    assert matched_row["units"] == 12


def test_date_2_date_range_inclusive(sales_df):
    """
    Step 7: Date Range Test
    Query: 'Show me all sales records from January 3, 2025 through January 8, 2025 inclusive.'
    Expected:
    - date >= 2025-01-03 AND date <= 2025-01-08
    - rows = 6
    - sales = 66,000
    - units = 105
    - average sales = 11,000
    - highest sales = 18,000
    - highest-sales date = 2025-01-06
    """
    query = "Show me all sales records from January 3, 2025 through January 8, 2025 inclusive."
    res = FilterEngine.execute(sales_df, query)

    assert res.matching_rows == 6, f"Expected 6 matching rows, got {res.matching_rows}"
    assert "date >= 2025-01-03" in res.filter_description
    assert "date <= 2025-01-08" in res.filter_description
    assert res.aggregations["sales"]["value"] == 66000.0
    assert res.aggregations["units"]["value"] == 105.0
    assert res.aggregations["sales_mean"]["value"] == 11000.0
    assert res.highest_record is not None
    assert res.highest_record["sales"] == 18000.0
    assert res.highest_record["date"] == "2025-01-06"


def test_date_3_inclusive_boundaries(sales_df):
    """
    Step 8: Inclusive Boundaries Test
    'from January 3 through January 8 inclusive'
    MUST include: January 3, 4, 5, 6, 7, 8
    MUST exclude: January 1, 2, 9, 10
    """
    query = "from January 3 through January 8, 2025 inclusive"
    res = FilterEngine.execute(sales_df, query)

    matched_dates = set(res.filtered_df["date"].tolist())
    expected_included = {"2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08"}
    expected_excluded = {"2025-01-01", "2025-01-02", "2025-01-09", "2025-01-10"}

    assert expected_included.issubset(matched_dates), f"Missing boundary dates: {expected_included - matched_dates}"
    for ex in expected_excluded:
        assert ex not in matched_dates, f"Date {ex} should have been excluded!"


def test_date_4_date_plus_numeric_filter(sales_df):
    """
    Step 9: Date + Numeric Filter Composition
    Query: 'Calculate total sales for records from January 3 through January 8, 2025 where discount is 5 or less.'
    Expected:
    - matching rows: Jan 3, Jan 6, Jan 8 (3 rows)
    - sales: 15,000 + 18,000 + 11,000 = 44,000
    - units: 12 + 14 + 25 = 51
    """
    query = "Calculate total sales and total units for records from January 3 through January 8, 2025 where discount is 5 or less."
    res = FilterEngine.execute(sales_df, query)

    assert res.matching_rows == 3, f"Expected 3 matching rows, got {res.matching_rows}"
    assert res.aggregations["sales"]["value"] == 44000.0
    assert res.aggregations["units"]["value"] == 51.0
    matched_dates = set(res.filtered_df["date"].tolist())
    assert matched_dates == {"2025-01-03", "2025-01-06", "2025-01-08"}


def test_date_5_date_plus_categorical_or(sales_df):
    """
    Step 10: Date + Categorical OR Composition
    Query: 'Calculate total sales from January 3 through January 8, 2025 where region is North or West.'
    Expected:
    - matching rows: Jan 3 (North: 15k), Jan 6 (West: 18k), Jan 8 (North: 11k) = 3 rows
    - sales = 44,000
    - units = 51
    """
    query = "Calculate total sales and total units from January 3 through January 8, 2025 where region is North or West."
    res = FilterEngine.execute(sales_df, query)

    assert res.matching_rows == 3, f"Expected 3 matching rows, got {res.matching_rows}"
    assert res.aggregations["sales"]["value"] == 44000.0
    assert res.aggregations["units"]["value"] == 51.0
    assert "date >= 2025-01-03" in res.filter_description
    assert "date <= 2025-01-08" in res.filter_description
    assert "region == 'North'" in res.filter_description or "North" in res.filter_description


def test_date_6_column_stored_as_string():
    """
    Step 13.6: Verify date filtering works when the date column is stored as str/object.
    """
    df_str = pd.DataFrame([
        {"date": "2025-01-01", "sales": 100},
        {"date": "2025-01-03", "sales": 200},
        {"date": "2025-01-05", "sales": 300},
    ])
    assert pd.api.types.is_string_dtype(df_str["date"]) or pd.api.types.is_object_dtype(df_str["date"])

    res = FilterEngine.execute(df_str, "records from January 3, 2025. Calculate total sales.")
    assert res.matching_rows == 1
    assert res.aggregations["sales"]["value"] == 200.0


def test_date_7_column_stored_as_datetime():
    """
    Step 13.7: Verify date filtering works when the date column is stored as pandas datetime64[ns].
    """
    df_dt = pd.DataFrame([
        {"date": pd.to_datetime("2025-01-01 10:30:00"), "sales": 100},
        {"date": pd.to_datetime("2025-01-03 14:15:00"), "sales": 200},
        {"date": pd.to_datetime("2025-01-05 09:00:00"), "sales": 300},
    ])
    assert pd.api.types.is_datetime64_any_dtype(df_dt["date"])

    res = FilterEngine.execute(df_dt, "records from January 3, 2025. Calculate total sales.")
    assert res.matching_rows == 1
    assert res.aggregations["sales"]["value"] == 200.0


def test_date_8_different_column_names():
    """
    Step 13.8: Verify detection of non-'date' column names (order_date, transaction_date, created_at).
    """
    # 1. order_date
    df_order = pd.DataFrame([
        {"order_date": "2025-01-01", "revenue": 500},
        {"order_date": "2025-01-03", "revenue": 1500},
    ])
    r1 = FilterEngine.execute(df_order, "records from January 3, 2025. Calculate total revenue.")
    assert r1.matching_rows == 1
    assert "order_date == 2025-01-03" in r1.filter_description
    assert r1.aggregations["revenue"]["value"] == 1500.0

    # 2. transaction_date
    df_tx = pd.DataFrame([
        {"transaction_date": "2025-01-03", "amount": 750},
        {"transaction_date": "2025-01-09", "amount": 850},
    ])
    r2 = FilterEngine.execute(df_tx, "records from January 3, 2025. Calculate total amount.")
    assert r2.matching_rows == 1
    assert "transaction_date == 2025-01-03" in r2.filter_description
    assert r2.aggregations["amount"]["value"] == 750.0


def test_date_9_invalid_date_expression():
    """
    Step 13.9: Verify that invalid date expressions do NOT silently fall back to all records.
    """
    df_sample = pd.DataFrame([{"date": "2025-01-01", "val": 10}, {"date": "2025-01-02", "val": 20}])
    res = FilterEngine.execute(df_sample, "records from InvalidMonth 99, 2025")
    assert res.matching_rows == 0
    assert "All Records" not in res.filter_description
    assert "could not parse" in res.filter_description or "could not" in res.filter_description.lower()


def test_date_10_ambiguous_multiple_date_columns():
    """
    Step 13.10: Ambiguous multiple date columns should trigger a clarification request rather than picking blindly.
    """
    df_multi = pd.DataFrame([
        {"order_date": ["2025-01-01"], "ship_date": ["2025-01-05"], "sales": [100]}
    ])
    res = FilterEngine.execute(df_multi, "records from January 3, 2025")
    assert res.matching_rows == 0
    assert "multiple date columns" in res.filter_description.lower() or "multiple temporal columns" in res.filter_description.lower()


# ==============================================================================
# Grouping / Ranking / Query Plan Tests (Bug #3 Steps 11 to 14)
# ==============================================================================

def test_grouping_1_total_sales_by_product(sales_df):
    """1. 'total sales by product' -> GROUP BY product, FILTER: none"""
    query = "total sales by product"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert plan.group_by == ["product"]

    res = FilterEngine.execute(sales_df, query)
    assert res.filter_description == "None"
    assert res.matching_rows == 10
    assert res.group_by == ["product"]


def test_grouping_2_total_sales_by_region(sales_df):
    """2. 'total sales by region' -> GROUP BY region, FILTER: none"""
    query = "total sales by region"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert plan.group_by == ["region"]

    res = FilterEngine.execute(sales_df, query)
    assert res.filter_description == "None"
    assert res.matching_rows == 10
    assert res.group_by == ["region"]


def test_grouping_3_total_sales_by_product_and_region(sales_df):
    """3. 'total sales by product and region' -> GROUP BY product, region, FILTER: none"""
    query = "total sales by product and region"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert plan.group_by == ["product", "region"]

    res = FilterEngine.execute(sales_df, query)
    assert res.filter_description == "None"
    assert res.matching_rows == 10
    assert res.group_by == ["product", "region"]


def test_grouping_4_show_every_product_region_combination(sales_df):
    """4. 'show every product-region combination' -> GROUP BY product, region, FILTER: none"""
    query = "show every product-region combination"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert plan.group_by == ["product", "region"]

    res = FilterEngine.execute(sales_df, query)
    assert res.filter_description == "None"
    assert res.matching_rows == 10
    assert res.group_by == ["product", "region"]


def test_grouping_5_rank_combinations_by_total_sales(sales_df):
    """5. 'rank combinations by total sales from highest to lowest' -> RANK/SORT, FILTER: none"""
    query = "rank combinations by total sales from highest to lowest"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert plan.ranking is not None
    assert plan.ranking["column"] == "sales"
    assert plan.ranking["direction"] == "desc"


def test_grouping_6_identify_highest_sales_combination(sales_df):
    """6. 'identify the highest sales combination' -> ARGMAX, FILTER: none"""
    query = "identify the highest sales combination"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert "highest" in plan.extremes


def test_grouping_7_calculate_average_sales_for_each_product(sales_df):
    """7. 'calculate average sales for each product' -> GROUP BY product + AVG(sales), FILTER: none"""
    query = "calculate average sales for each product"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is None
    assert plan.group_by == ["product"] or any(sec["group_by"] == ["product"] for sec in plan.secondary_analysis)

    res = FilterEngine.execute(sales_df, query)
    assert res.filter_description == "None"
    assert res.matching_rows == 10


def test_grouping_8_calculate_total_sales_where_region_is_north(sales_df):
    """8. 'calculate total sales where region is North' -> FILTER region == North"""
    query = "calculate total sales where region is North"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is not None
    assert "region == 'North'" in plan.filter.to_expression()

    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 3
    assert res.aggregations["sales"]["value"] == 38000.0


def test_grouping_9_calculate_total_sales_where_region_is_north_or_west(sales_df):
    """9. 'calculate total sales where region is North or West' -> FILTER region IN [North, West]"""
    query = "calculate total sales where region is North or West"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is not None
    expr = plan.filter.to_expression()
    assert "North" in expr and "West" in expr

    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 5
    assert res.aggregations["sales"]["value"] == 76000.0


def test_grouping_10_filter_plus_grouping(sales_df):
    """10. 'calculate total sales for each product where region is North' -> FILTER: region == North, GROUP BY: product"""
    query = "calculate total sales for each product where region is North"
    plan = FilterEngine.parse_query_plan(query, list(sales_df.columns), sales_df)
    assert plan.filter is not None
    assert "region == 'North'" in plan.filter.to_expression()
    assert plan.group_by == ["product"] or any(sec["group_by"] == ["product"] for sec in plan.secondary_analysis)

    res = FilterEngine.execute(sales_df, query)
    assert res.matching_rows == 3
    assert "region == 'North'" in res.filter_description


def test_grouping_12_full_original_query(sales_df):
    """
    Step 12: Full Original Query Verification
    Query:
    'Analyze total sales and total units by product and region. Show every product-region combination,
    rank the combinations by total sales from highest to lowest, and identify the highest and lowest sales combination.
    Also calculate the average sales for each product. Do not make causal claims.'
    """
    query = (
        "Analyze total sales and total units by product and region. "
        "Show every product-region combination, rank the combinations by total sales from highest to lowest, "
        "and identify the highest and lowest sales combination. "
        "Also calculate the average sales for each product. Do not make causal claims."
    )
    res = FilterEngine.execute(sales_df, query)

    # 1. No filter applied, operates on all 10 records
    assert res.filter_description == "None"
    assert res.matching_rows == 10
    assert res.total_rows == 10
    assert res.aggregations["sales"]["value"] == 113500.0
    assert res.aggregations["units"]["value"] == 169.0

    # 2. Group by product and region
    assert res.group_by == ["product", "region"]
    assert res.grouped_records is not None
    assert len(res.grouped_records) == 6

    # 3. Product + Region totals and ranking
    r = res.grouped_records
    assert r[0]["product"] == "Laptop" and r[0]["region"] == "West" and r[0]["sales"] == 38000 and r[0]["units"] == 30
    assert r[1]["product"] == "Laptop" and r[1]["region"] == "North" and r[1]["sales"] == 27000 and r[1]["units"] == 22
    assert r[2]["product"] == "Phone" and r[2]["region"] == "South" and r[2]["sales"] == 17000 and r[2]["units"] == 42
    assert r[3]["product"] == "Tablet" and r[3]["region"] == "East" and r[3]["sales"] == 13000 and r[3]["units"] == 32
    assert r[4]["product"] == "Phone" and r[4]["region"] == "North" and r[4]["sales"] == 11000 and r[4]["units"] == 25
    assert r[5]["product"] == "Tablet" and r[5]["region"] == "South" and r[5]["sales"] == 7500 and r[5]["units"] == 18

    # 4. Extremes
    assert res.highest_record is not None
    assert res.highest_record["combination"] == "Laptop / West"
    assert res.highest_record["sales"] == 38000.0

    assert res.lowest_record is not None
    assert res.lowest_record["combination"] == "Tablet / South"
    assert res.lowest_record["sales"] == 7500.0

    # 5. Average sales by product
    assert res.secondary_results is not None
    sec = res.secondary_results[0]
    assert sec["group_by"] == "product"
    sec_map = {rec["product"]: rec["sales"] for rec in sec["records"]}
    assert abs(sec_map["Laptop"] - 16250.0) < 0.01
    assert abs(sec_map["Phone"] - 9333.33) < 0.1
    assert abs(sec_map["Tablet"] - 6833.33) < 0.1


def test_grouping_14_arbitrary_dataset_independence():
    """
    Step 14: Arbitrary Dataset Independence
    Verify parser works on arbitrary schemas without hardcoded column names or values.
    """
    custom_df = pd.DataFrame([
        {"customer_tier": "Gold", "country": "Canada", "order_value": 500, "items_purchased": 2},
        {"customer_tier": "Gold", "country": "Canada", "order_value": 700, "items_purchased": 3},
        {"customer_tier": "Silver", "country": "USA", "order_value": 300, "items_purchased": 1},
        {"customer_tier": "Silver", "country": "USA", "order_value": 400, "items_purchased": 2},
    ])
    q = "Analyze total order_value and total items_purchased by customer_tier and country. Rank the combinations by order_value from highest to lowest."
    res = FilterEngine.execute(custom_df, q)

    assert res.filter_description == "None"
    assert res.matching_rows == 4
    assert res.group_by == ["customer_tier", "country"]
    assert res.grouped_records is not None
    assert len(res.grouped_records) == 2
    top = res.grouped_records[0]
    assert top["customer_tier"] == "Gold" and top["country"] == "Canada"
    assert top["order_value"] == 1200
    assert top["items_purchased"] == 5


