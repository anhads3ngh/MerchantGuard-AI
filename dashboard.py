"""Optional local Streamlit control room for synthetic data only."""
import streamlit as st
import pandas as pd
import plotly.express as px
from merchantguard.store import connect
from merchantguard.model import load_metadata

st.set_page_config(page_title='MerchantGuard | Risk Operations',page_icon='🛡️',layout='wide')
st.markdown('### 🛡️ MERCHANTGUARD / RISK OPERATIONS')
st.caption('SYNTHETIC DATA  ·  Research prototype  ·  No payment decisions are automated')
with connect() as conn:
    daily=pd.read_sql_query("SELECT substr(event_time,1,10) AS day,COUNT(*) AS transactions,ROUND(SUM(amount),2) AS volume, SUM(fraud_label) AS simulated_fraud FROM transactions GROUP BY day ORDER BY day",conn)
    merchants=pd.read_sql_query("SELECT merchant_id,merchant_category,COUNT(*) as transactions,ROUND(SUM(amount),2) as volume, ROUND(AVG(fraud_label)*100,2) AS synthetic_fraud_pct FROM transactions GROUP BY merchant_id,merchant_category ORDER BY volume DESC LIMIT 20",conn)
    preds=pd.read_sql_query("SELECT score,needs_review FROM predictions",conn)
    alerts=pd.read_sql_query("SELECT created_at, feature_name, psi, severity FROM alerts ORDER BY alert_id DESC LIMIT 20",conn)
try:
    metrics=load_metadata()['test_metrics']
except FileNotFoundError:
    metrics={}
if daily.empty:
    st.info('Run `python -m merchantguard demo` from a terminal first.')
    st.stop()
a,b,c,d=st.columns(4)
a.metric('Synthetic transactions', f'{int(daily.transactions.sum()):,}')
b.metric('Processed volume (CAD-like units)', f'{daily.volume.sum():,.0f}')
c.metric('Test PR-AUC', f"{metrics.get('pr_auc',0):.3f}")
d.metric('Review alerts', f'{len(alerts):,}')
left,right=st.columns([1.5,1])
with left:
    st.plotly_chart(px.area(daily,x='day',y='transactions',title='Transaction throughput over time',template='plotly_white'),use_container_width=True)
with right:
    if not preds.empty:
        st.plotly_chart(px.histogram(preds,x='score',nbins=35,title='Held-out payment risk scores',template='plotly_white'),use_container_width=True)
st.markdown('#### Merchant performance')
st.dataframe(merchants,use_container_width=True,hide_index=True)
st.markdown('#### Monitoring alerts')
if alerts.empty:st.success('No drift alerts logged')
else:st.dataframe(alerts,use_container_width=True,hide_index=True)
st.caption('All merchants, labels and transaction amounts are generated; displayed metrics are not real business outcomes.')
