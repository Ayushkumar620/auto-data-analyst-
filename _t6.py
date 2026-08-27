from app import app
import io, traceback
app.config['TESTING']=True
c=app.test_client()
payload=open('uploads/real_world_analyst_stress_test.csv','rb').read()
cmds=[
 'chart by client_segment','chart net_profit','line chart net_profit','show net profit over time',
 'forecast net_profit','forecast units_sold','predict net_profit','predict customer_score',
 'total gross_sales','average net_profit','max net_profit','min gross_sales',
 'top product_group by gross_sales','net_profit by product_group','sum gross_sales by sales_channel',
 'compare net_profit across product_group','how does net_profit vary by market_zone',
 'gross_sales grouped by risk_level','average of net_profit for each client_segment',
 'insights','report','anomalies','clean','describe','correlation'
]
for cmd in cmds:
    try:
        r=c.post('/api/analyze', data={'file':(io.BytesIO(payload),'real_world_analyst_stress_test.csv'),'command':cmd}, content_type='multipart/form-data')
        body=r.get_data(as_text=True)
        flag='NOJSON' if ('json' not in (r.content_type or '')) else 'json'
        bad = 'Analysis failed' in body
        print(r.status_code, flag, 'BAD' if bad else '  ', '|', cmd)
    except Exception as e:
        print('EXC', cmd, '=>', type(e).__name__, str(e)[:100])
