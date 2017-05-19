import numpy as np
import pandas as pd
import sys

def one_hot_encode(df, column):
    for attr in df[column].unique():
        binary_conversion = lambda x: int(x == attr)
        df[attr] = df[column].apply(binary_conversion)
    df = df.drop(column, axis=1)
    return df

def convert_to_binary(df, column, attr):
    convert = lambda x: int(x == attr)
    df[column] = df[column].apply(convert)
    return df

# def main(argv):
if len(sys.argv) != 2:
    print "Usage python parse_dataset <filename>"
    sys.exit(2)

FILENAME = sys.argv[1]

cancer_df = pd.read_csv(FILENAME, '\t')

# # print column names
# print cancer_df.columns

# # print all patient ids
# print cancer_df.PATIENT_ID

# drop unnecessary columns
cancer_df = cancer_df.drop('OS_MONTHS', axis=1).drop('OS_STATUS', axis=1,).drop('INTCLUST', axis=1).drop('COHORT', axis=1).drop('LATERALITY', axis=1).drop('HISTOLOGICAL_SUBTYPE', axis=1).drop('THREEGENE', axis=1)

# remove any instances with missing data
cancer_df = cancer_df.dropna(axis=0, how='any')
cancer_df = cancer_df[cancer_df.BREAST_SURGERY != 'null']
cancer_df = cancer_df[cancer_df.CELLULARITY != 'null']
cancer_df = cancer_df[cancer_df.HER2_SNP6 != 'UNDEF']

# convert textual data to numerical
# YES = 1, NO = 0
therapies = ['CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY']
for therapy in therapies:
    cancer_df = convert_to_binary(cancer_df, therapy, 'YES')

# pos = 1, neg = 0
cancer_df = convert_to_binary(cancer_df, 'ER_IHC', 'pos')

# post = 1, pre = 0
cancer_df = convert_to_binary(cancer_df, 'INFERRED_MENOPAUSAL_STATE', 'post')

# MASTECTOMY = 1, BREAST CONSERVING = 0
cancer_df = convert_to_binary(cancer_df, 'BREAST_SURGERY', 'MASTECTOMY')

# one hot encode categorical data
categorical_data = ['VITAL_STATUS', 'CELLULARITY', 'HER2_SNP6', 'CLAUDIN_SUBTYPE']
for cat in categorical_data:
    cancer_df = one_hot_encode(cancer_df, cat)

# represent data and target labels in matrix form
X = cancer_df.drop('PATIENT_ID', axis=1).drop(therapies, axis=1).as_matrix()
Y = cancer_df[therapies].as_matrix()