#%%
import pandas as pd
df = pd.read_csv('MOESM3_ESM_seqs.csv')
# %%
from mrna_utils import *
import numpy as np
from streaming import MDSWriter

import pandas as pd
import gc
import glob
import os
import shutil
import re
import tqdm

#%%

mrnatok = MRNATOK()
#%% check if tok work
def check_tokenization(idx, df):
    idx_check = 0
    five = df["5' UTR"].iloc[idx_check]
    codons = df["CDS"].iloc[idx_check]
    three = df["3' UTR"].iloc[idx_check]
    ret = mrnatok.get_ids_given_three_seg(five, three, codons)

    str_ref = five+codons+three
    str_comp = ''.join([mrnatok.map_id_tok[ri] for ri in ret])
    str_comp = str_comp.replace('<5b>','')
    str_comp = str_comp.replace('<3b>','')
    str_comp = str_comp.replace('<3e>','')
    str_comp = str_comp.replace('<cb>','')

    return (str_ref.upper().replace('T', 'U')==str_comp)

# %% clean dirty sequence
df_keep_5 = (~df["5' UTR"].isnull())
df_keep_3 = (~df["3' UTR"].isnull())
df_keep_cds = (~df["CDS"].isnull())
df_keep = df_keep_5 | df_keep_3 | df_keep_cds
df_keep = df_keep_cds
df_keep.sum()
df_seq = df[df_keep].copy()
df_seq['is_codon_valid'] = df_seq['CDS'].map(mrnatok.validate_codon) # 0 for OK, other
df_seq = df_seq[df_seq['is_codon_valid']==0].copy()

df_seq_pure = df_seq.loc[:, ["5' UTR","3' UTR","CDS"]].fillna('')
#%%
cns_target =  ['half-life (PC1)',
       'Bazzini_ActD_HEK293_1', 'Bazzini_ActD_HeLa_1', 'Bazzini_ActD_RPE_1',
       'Bazzini_4sU_K562_1', 'Akimitsu_BrU_HeLa_1', 'Rinn_ActD_K562_1',
       'Rinn_ActD_K562_2', 'Rinn_ActD_K562_3', 'Rinn_ActD_H1ESC_1',
       'Rinn_ActD_H1ESC_2', 'Rinn_ActD_H1ESC_3', 'Akimitsu2_BrU4sU_HeLa_1',
       'Jaffrey_ActD_HEK293_1', 'Cramer_4sU_K562_1', 'Shendure_4sU_A549_1',
       'Cramer2_4sU_K562_1', 'Oberdoerffer_BrU_HeLa_1',
       'Dieterich_4sU_HEK293_1', 'Dieterich_4sU_HEK293_2',
       'Dieterich_4sU_MCF7_1', 'Dieterich_4sU_MCF7_2', 'He_ActD_HeLa_1',
       'He_ActD_HeLa_2', 'Marks_ActD_HeLa_1', 'Darnell_ActD_HepG2_1',
       'Mortazavi_4sU_GM12878_1', 'Rissland2_ActD_HEK293_1',
       'Rissland2_ActD_HEK293_2', 'Rissland2_Aman_HEK293_1',
       'Rissland2_Aman_HEK293_2', 'Rissland2_4sU_HEK293_1',
       'Rissland2_4sU_HEK293_2', 'Zimmer_ActD_Bcell_1', 'Gejman_4sU_GM07029_1',
       'Gejman_4sU_GM07029_2', 'Gejman_4sU_GM07029_3', 'Gejman_4sU_GM10835_1',
       'Gejman_4sU_GM10835_2', 'Gejman_4sU_GM10835_3', 'Gejman_4sU_GM12813_1',
       'Gejman_4sU_GM12813_2', 'Gejman_4sU_GM12813_3', 'Gejman_4sU_GM12813_4',
       'Gejman_4sU_GM12813_5', 'Gejman_4sU_GM07019_1', 'Gejman_4sU_GM12812_1',
       'Gejman_4sU_GM12814_1', 'Gejman_4sU_GM12815_1', 'Simon_4sU_K562_1',
       'Simon_4sU_K562_2', 'Rissland_4sU_HEK293_1', 'Rissland_4sU_HEK293_2',
       'Rissland_4sU_HEK293_3', 'Rissland_4sU_HEK293_4']
df_target = df_seq.loc[:, cns_target]

# %%
# import seaborn as sns
# sns.pairplot(df_target.sample(1000))
df_target.to_excel('mrna_downstream_cleansed.xlsx', index=False)
# %%
from sklearn.preprocessing import StandardScaler
myn = StandardScaler()
tgt_after_transform = myn.fit_transform(df_target.values)

from joblib import dump, load
dump(myn, 'targets_stand.joblib') 
#%%
import numpy as np
N = df_target.shape[0]
idx = np.arange(N)
from sklearn.model_selection import train_test_split
idx_train, idx_te = train_test_split(idx, test_size=1000, random_state=42)
idx_tr, idx_va = train_test_split(idx_train, test_size=1000, random_state=42)
np.savez_compressed('idx_tr_va_te.npz', idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)
# %%
vocab_size = 4096

fn_tok = f"../mrna_llm/lit-gpt/tok_wordpiece_{vocab_size}.json"
offset = 0x4e00
unk = '[UNK]'

def get_lst_str(idx_cur):
    for iii, ii in tqdm.tqdm(enumerate(idx_cur)):
        df_seq_i = df_seq_pure.iloc[ii, :]
        five, three, codons = df_seq_i["5' UTR"], df_seq_i["3' UTR"], df_seq_i["CDS"]
        cret = mrnatok.get_ids_given_three_seg(five, three, codons)
        str_in = ''.join([chr(xy+offset) for xy in cret])
        yield str_in



def process_dir(idx_cur, path_cur):
    data_out_path = path_cur
    shutil.rmtree(data_out_path, ignore_errors=True)
    print(data_out_path)
    columns = {'targets':'pkl', 'ids':'pkl'}

    
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    # tokenizer = Tokenizer(models.WordPiece(max_input_chars_per_word=20000))
    from transformers import PreTrainedTokenizerFast, AutoTokenizer, WordpieceTokenizer
    tok = Tokenizer.from_file(fn_tok)
    # tok.add_special_tokens([unk])
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token=unk
    )

    lst_str = list(get_lst_str(idx_cur))
    
    print('----- start of tok ----')
    after_tok = wrapped_tokenizer(lst_str)
    print('----- end of tok ----')

    
    shutil.rmtree(data_out_path, ignore_errors=True)
    with MDSWriter(out=data_out_path, columns=columns, compression=None, keep_local=True) as out:
        for iii, ii in tqdm.tqdm(enumerate(idx_cur)):
            c_target = tgt_after_transform[ii,]
            sample = {
                'ids': np.array(after_tok[iii].ids, dtype=np.uint16),
                'targets': c_target
            }
            out.write(sample)

    print('END of writing')


#%%
# process_dir(idx_tr, f'./mds_wordpiece_{vocab_size}/tr/') # 11089
# process_dir(idx_va, f'./mds_wordpiece_{vocab_size}/va/') # 11089
process_dir(idx_te, f'./mds_wordpiece_{vocab_size}/te/') # 11089


# %%
