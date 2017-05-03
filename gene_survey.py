import numpy as np
import pandas as pd

common_genes = ["HSP90", "IDO", "DNA", "BCLXL", "BCL2", "CDK", "TGFB2", "PARP1", "VEGF", "PI3K", "COX2", "EGFR",
"HTERT", "BCRABL", "PDGFR", "DHFR", "CHK1", "MTOR", "RAR", "ERBB2"]

CANCER = "Cancer/"
BLADDER = "Bladder/blca/tcga/data_methylation_hm450.txt"
BREAST = "Breast/brca/data_expression.txt"
RECTAL = "Colorectal/coadread/tcga/data_methylation_hm450.txt"
ESOPHOGEAL = "Esophageal/esca/tcga/data_methylation_hm450.txt"
KIDNEY1 = "Kidney Renal/kich/tcga/data_methylation_hm450.txt"
KIDNEY2 = "Kidney Renal/kirc/tcga/data_methylation_hm450.txt"
SKIN = "Skin Cancer/skcm/tcga/data_methylation_hm450.txt"
TEST = "Testicular/tgct/tcga/data_methylation_hm450.txt"
UTERINE1 = "Uterine/ucs/tcga/data_methylation_hm450.txt"
UTERINE2 = "Uterine/ucec/tcga/data_methylation_hm450.txt"

all_datasets = [BLADDER, BREAST, RECTAL, ESOPHOGEAL, KIDNEY1, KIDNEY2, SKIN, TEST, UTERINE1, UTERINE2]

gene_survey = []
sample_size = 0

for data in all_datasets:
    df = pd.read_csv(CANCER+data, sep='\t')
    sample_size += len(df.columns)

    # format Hugo Symbols
    df.Hugo_Symbol = df.Hugo_Symbol.str.upper()

    genes = map(lambda x: x in df.Hugo_Symbol.tolist(), common_genes)
    gene_survey.append(np.array(genes))

gene_survey_df = pd.DataFrame(gene_survey, columns=common_genes)






