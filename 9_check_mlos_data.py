#%%
import pandas as pd
fn = 'MLOSx-FlubCDG-correctedExpressions-mRNA-seqs-20231205 1.csv'
df = pd.read_csv(fn)
# %%
df['mlos'].value_counts()
# %%
df['expression_relative_QCadjusted'].hist(bins=20)
# %%
df['len_5'] = df['utr5'].map(len)
df['len_3'] = df['utr3'].map(len)
df['len_cds'] = df['CDS'].map(len)

# %%
