from agent.command_parser import CommandParser
import pandas as pd
cols=['recorded_on','client_segment','market_zone','product_group','sales_channel','discount_rate','gross_sales','operating_cost','net_profit','marketing_spend','conversion_rate','risk_level','transaction_ref','sparse_notes','cost_adjustment']
df=pd.read_csv('uploads/real_world_analyst_stress_test.csv')
p=CommandParser(df)
fail=0
for c in cols:
    for q in [f'average {c}', f'mean of {c}', f'{c} by market_zone', f'chart {c}', f'forecast target={c}', f'predict target={c}']:
        try:
            p.parse(q)
        except Exception as e:
            fail+=1
            print('CRASH', q, '=>', type(e).__name__, ':', str(e)[:90])
print('total fails', fail)
