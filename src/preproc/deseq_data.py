import pandas as pd
from os import path
from sys import platform

def load():
    prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
    data_folder = path.join(prefix, 'Shared drives/NIAAA_ASSIST/Data')
    
    deseq = pd.read_excel(data_folder + '/deseq.alc.vs.control.age.rin.batch.gender.PMI.corrected.w.prot.coding.gene.name.xlsx')
    deseq['abs_log2FC'] = abs(deseq['log2FoldChange'])
    return deseq

deseq = load()