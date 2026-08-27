import pandas as pd, traceback
from agent.command_parser import CommandParser
df = pd.read_csv('uploads/real_world_analyst_stress_test.csv')
parser = CommandParser(df)
cmds = ['summary','insights','anomalies','chart','chart by client_segment','net profit by product_group','average discount_rate','gross sales by market_zone','forecast target=net_profit','predict target=net_profit','what is the average gross_sales','total gross_sales by product_group','chart by product_group']
for cmd in cmds:
    try:
        r = parser.parse(cmd)
        print('OK ', cmd, '=>', str(r)[:60].replace(chr(10),' '))
    except Exception as e:
        print('EXC', cmd, '=>', type(e).__name__, e)
