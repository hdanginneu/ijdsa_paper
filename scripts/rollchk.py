import sys,pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv(sys.argv[1])
dc=[c for c in d.columns if 'date' in c.lower()][0]; pc=[c for c in d.columns if c.lower() in ('close','price')][0]
d[dc]=pd.to_datetime(d[dc]); d=d.sort_values(dc).set_index(dc)
p=pd.to_numeric(d[pc].astype(str).str.replace(',',''),errors='coerce')
r=np.log(p).diff().dropna(); idx=r.index
rolls=[]
for y in range(2008,2027):
  for m in (1,3,5,7,9,11):
    days=idx[(idx.year==y)&(idx.month==m)]
    if len(days)<6: continue
    ltd=days[-5]
    wk=ltd-pd.Timedelta(days=ltd.weekday())
    cand=idx[idx>=wk]
    if len(cand): rolls.append(cand[0])
rolls=pd.DatetimeIndex(sorted(set(rolls))); a=r.abs(); isroll=a.index.isin(rolls)
print("cols",dc,pc,"| roll days",isroll.sum(),"other",(~isroll).sum())
print("mean|r| roll %.4f other %.4f | median roll %.4f other %.4f"%(a[isroll].mean(),a[~isroll].mean(),a[isroll].median(),a[~isroll].median()))
print("MW one-sided p %.3f"%stats.mannwhitneyu(a[isroll],a[~isroll],alternative='greater').pvalue)
q=a[~isroll].quantile([.95,.99]).values
print("share roll>p95 other %.3f, >p99 %.3f (expected .05/.01)"%((a[isroll]>q[0]).mean(),(a[isroll]>q[1]).mean()))
m=a.index>=pd.Timestamp("2020-12-01")
print("eval: roll n",(isroll&m).sum(),"mean roll %.4f other %.4f MW p %.3f"%(a[isroll&m].mean(),a[(~isroll)&m].mean(),stats.mannwhitneyu(a[isroll&m],a[(~isroll)&m],alternative='greater').pvalue))
print("largest roll-day |r| %.4f on %s"%(a[isroll].max(),a[isroll].idxmax().date()))
