"""EDA de UNSW-NB15. Ejecutar desde la raíz: python src/eda.py"""
import pandas as pd
pd.set_option("display.width",200); pd.set_option("display.max_columns",60); pd.set_option("display.max_rows",100)
tr=pd.read_csv("UNSW_NB15_training-set.csv"); te=pd.read_csv("UNSW_NB15_testing-set.csv")
print(tr.shape, te.shape)
print("cols iguales:", list(tr.columns)==list(te.columns))
print(tr.dtypes.value_counts())
print(list(tr.columns))
for n,d in [("train",tr),("test",te)]:
    print("\n==",n)
    print("label:\n", d.label.value_counts().to_frame().assign(pct=lambda x:(x['count']/len(d)*100).round(2)))
    print("attack_cat:\n", d.attack_cat.value_counts().to_frame().assign(pct=lambda x:(x['count']/len(d)*100).round(2)))
    print("nulos totales:", int(d.isna().sum().sum()))
    print("duplicados (sin id):", int(d.drop(columns="id").duplicated().sum()))
    print("duplicados (sin id, label, attack_cat):", int(d.drop(columns=["id","label","attack_cat"]).duplicated().sum()))
    print("id min/max/unico:", d.id.min(), d.id.max(), d.id.is_unique)
cat=tr.select_dtypes(include="str").columns.tolist(); print("\ncategóricas:",cat)
for c in cat:
    print(c, "train",tr[c].nunique(),"test",te[c].nunique(), "solo en test:", sorted(set(te[c])-set(tr[c]))[:10], "solo en train:", sorted(set(tr[c])-set(te[c]))[:10])
print(tr.proto.value_counts().head(8)); print(tr.service.value_counts()); print(tr.state.value_counts())
# id vs label (el orden del archivo)
tr["blk"]=pd.cut(tr.id,10); print(tr.groupby("blk",observed=True).label.mean().round(3))
te["blk"]=pd.cut(te.id,10); print(te.groupby("blk",observed=True).label.mean().round(3))

# ---- parte 2 ----
feat=[c for c in tr.columns if c not in("id","label","attack_cat")]
# duplicados por clase
d=tr.duplicated(subset=feat,keep="first"); print("dup feat train por label:\n", tr.assign(d=d).groupby("label").d.agg(["sum","mean"]).round(3))
# conflictos: mismas features, distinto label
g=tr.groupby(feat,dropna=False).label.nunique(); print("grupos de feats con labels conflictivos (train):",int((g>1).sum()))
g2=tr.groupby(feat).attack_cat.nunique(); print("grupos con attack_cat conflictiva:",int((g2>1).sum()))
# solapamiento train/test
m=te.merge(tr[feat].drop_duplicates(),on=feat,how="left",indicator=True)
print("filas de test cuyas features exactas ya están en train:", int((m._merge=="both").sum()), f"({(m._merge=='both').mean():.1%})")
# constantes / casi constantes
nun=tr[feat].nunique().sort_values(); print("nunique bajo:\n", nun.head(12))
# estadísticos numéricos
num=tr[feat].select_dtypes("number")
desc=num.describe().T[["mean","50%","max"]]; desc["skew"]=num.skew(); desc["pct0"]=(num==0).mean()
print(desc.sort_values("skew",ascending=False).round(2).to_string())
# correlación con label
c=num.corrwith(tr.label).abs().sort_values(ascending=False); print("top corr |label|:\n",c.head(10).round(3))
# pares muy correlacionados
cm=num.corr().abs(); pairs=[(a,b,cm.loc[a,b]) for i,a in enumerate(cm.columns) for b in cm.columns[i+1:] if cm.loc[a,b]>0.95]
print("pares |r|>0.95:",len(pairs)); [print(f"  {a}~{b}: {v:.3f}") for a,b,v in sorted(pairs,key=lambda x:-x[2])]
# sttl/ct_state_ttl: tasa de ataque
for c in ["sttl","ct_state_ttl"]: print(tr.groupby(c).label.agg(["mean","size"]).round(3).head(12))
print(tr.groupby("service").label.mean().round(3))
print(tr.groupby("proto").label.agg(["mean","size"]).sort_values("size",ascending=False).head(6).round(3))
# desplazamiento train vs test (medias estandarizadas)
shift=((te[num.columns].mean()-num.mean())/num.std()).abs().sort_values(ascending=False); print("mayor shift train->test (en sd):\n",shift.head(8).round(2))
