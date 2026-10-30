import sys, numpy as np, pandas as pd
from scipy import stats
d=pd.read_csv(sys.argv[1],parse_dates=['date']).sort_values('date').set_index('date')
P=d['close'].astype(float); ret=np.log(P).diff()
sig=np.sqrt((ret**2).ewm(alpha=0.06,adjust=False).mean())
dates=P.index; pos={t:i for i,t in enumerate(dates)}
B=[("B1","2020-12-01","2021-11-25","2019-11-27","2020-11-20"),("B2","2021-11-26","2022-11-24","2020-11-23","2021-11-17"),
   ("B3","2022-11-25","2023-11-22","2021-11-18","2022-11-16"),("B4","2023-11-23","2024-11-18","2022-11-17","2023-11-14"),
   ("B5","2024-11-19","2026-01-27","2023-11-15","2024-11-08"),("B6","2026-01-28","2026-09-29","2025-01-23","2026-01-19")]
def q(e,p):
    e=np.sort(e); n=len(e); k=int(np.ceil((n+1)*p))-1 if p>=.5 else int(np.floor((n+1)*p))-1
    return e[min(max(k,0),n-1)]
def rows(a,b,h):
    o=[t for t in dates if pd.Timestamp(a)<=t<=pd.Timestamp(b) and pos[t]+h<len(dates)]
    i=np.array([pos[t] for t in o]); return i, np.log(P.values[i+h]/P.values[i])
L=0.9; out=[]; hits={m:[] for m in ('conf','vol','heur')}
for blk,ta,tb,ca,cb in B:
  ic,_=rows(ca,cb,1); rmse1=np.sqrt(np.mean((P.values[ic+1]-P.values[ic])**2))
  for h in range(1,8):
    ic,ec=rows(ca,cb,h); it,et=rows(ta,tb,h)
    p0=P.values[it]; y=P.values[it+h]
    lo={'conf':p0*np.exp(q(ec,(1-L)/2)),
        'vol':p0*np.exp(q(ec/(sig.values[ic]*np.sqrt(h)),(1-L)/2)*sig.values[it]*np.sqrt(h)),
        'heur':p0-stats.norm.ppf((1+L)/2)*rmse1*np.sqrt(h)}
    for m,l in lo.items():
      br=y<l; x=int(br.sum()); f=stats.binom.cdf(x,len(br),.05)
      z='green' if f<.95 else 'yellow' if f<.9999 else 'red'
      out.append(dict(block=blk,h=h,method=m,n=len(br),breaches=x,zone=z,short=np.mean((l-y)[br]) if x else np.nan,cap=np.mean(p0-l)))
      if h==1: hits[m].append((~br).astype(int))
o=pd.DataFrame(out)
print(o.groupby(['method','zone']).size().unstack(fill_value=0))
for h in (1,5):
  g=o[o.h==h].groupby('method').apply(lambda x: pd.Series(dict(rate=x.breaches.sum()/x.n.sum(),short=np.nansum(x.short*x.breaches)/x.breaches.sum(),cap=np.sum(x.cap*x.n)/x.n.sum(),red=(x.zone=='red').sum(),yellow=(x.zone=='yellow').sum())))
  print('h',h); print(g.round(3))
# Christoffersen pooled, lower bound coverage 0.95, no cross-block transitions
def ll(k,n,p): return (n-k)*np.log(1-p)+k*np.log(p) if 0<p<1 else 0.0
for m,segs in hits.items():
  miss=[1-s for s in segs]; n=sum(len(s) for s in miss); k=sum(s.sum() for s in miss)
  LRuc=-2*(ll(k,n,.05)-ll(k,n,k/n))
  n00=n01=n10=n11=0
  for s in miss:
    a,b=s[:-1],s[1:]; n00+=np.sum((a==0)&(b==0)); n01+=np.sum((a==0)&(b==1)); n10+=np.sum((a==1)&(b==0)); n11+=np.sum((a==1)&(b==1))
  p01=n01/(n00+n01); p11=n11/(n10+n11) if n10+n11 else 0; p2=(n01+n11)/(n00+n01+n10+n11)
  LRind=-2*(ll(n01+n11,n00+n01+n10+n11,p2)-(ll(n01,n00+n01,p01)+ll(n11,n10+n11,p11)))
  LRcc=LRuc+LRind
  print(m,'misses',k,'n01',n01,'n11',n11,'p_uc %.4f p_ind %.4f p_cc %.4f'%(1-stats.chi2.cdf(LRuc,1),1-stats.chi2.cdf(LRind,1),1-stats.chi2.cdf(LRcc,2)))
