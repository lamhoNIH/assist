import os
import pandas as pd
from .input import Input

class DESeqData:
    __deseq = None
    
    def get_deseq():
        if DESeqData.__deseq is None:
            DESeqData()
        return DESeqData.__deseq
    
    def __init__(self):
        root_dir = Input.getPath()
        DESeqData.__deseq = pd.read_excel(os.path.join(root_dir, 'deseq.alc.vs.control.age.rin.batch.gender.PMI.corrected.w.prot.coding.gene.name.xlsx'))
        DESeqData.__deseq['abs_log2FC'] = abs(DESeqData.__deseq['log2FoldChange'])