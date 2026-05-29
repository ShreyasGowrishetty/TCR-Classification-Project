import pandas as pd
import numpy as np
import re

LABEL_MAP = {
    # VIRAL
    'Influenza':                                'viral',
    'Cytomegalovirus (CMV)':                    'viral',
    'Epstein Barr virus (EBV)':                 'viral',
    'Human immunodeficiency virus (HIV)':       'viral',
    'Yellow fever virus':                       'viral',
    'Hepatitis C virus':                        'viral',
    'COVID-19':                                 'viral',
    'Herpes simplex virus 2 (HSV2)':            'viral',
    'SARS-CoV-2':                               'viral',
    'Herpes simplex virus (HSV)':               'viral',
    'Herpes simplex virus 1 (HSV1)':            'viral',
    'Vaccinia virus':                           'viral',
    'Dengue virus':                             'viral',
    'West Nile virus':                          'viral',
    'Hepatitis B virus':                        'viral',
    'Human papillomavirus (HPV)':               'viral',
    'Adenovirus':                               'viral',
    'Parvovirus B19':                           'viral',
    'HTLV-1':                                   'viral',  # virus-caused; TCR recognizes viral Tax antigen

    # BACTERIAL
    'M. tuberculosis':                          'bacterial',
    'ARDS':                                     'bacterial',  # majority ARDS TCRs in VDJdb are Mtb-reactive MAIT cells
    'Lyme disease':                             'bacterial',
    'Staphylococcus aureus':                    'bacterial',
    'Salmonella':                               'bacterial',
    'Yersinia':                                 'bacterial',
    'Chlamydia':                                'bacterial',
    'Mycobacterium':                            'bacterial',

    # CANCER
    'Melanoma':                                 'cancer',
    'Tumor associated antigen (TAA)':           'cancer',
    'Colorectal cancer':                        'cancer',
    'Lung cancer':                              'cancer',
    'Breast cancer':                            'cancer',
    'Ovarian cancer':                           'cancer',
    'Prostate cancer':                          'cancer',
    'Leukemia':                                 'cancer',
    'Lymphoma':                                 'cancer',
    'Glioblastoma':                             'cancer',
    'Renal cell carcinoma':                     'cancer',
    'Pancreatic cancer':                        'cancer',
    'Bladder cancer':                           'cancer',
    'Hepatocellular carcinoma':                 'cancer',
    'Cervical cancer':                          'cancer',
    'Gastric cancer':                           'cancer',
    'Esophageal cancer':                        'cancer',
    'Head and neck cancer':                     'cancer',
    'NY-ESO-1':                                 'cancer',

    # AUTOIMMUNE
    'Allergy':                                  'autoimmune',
    'Alzheimer\'s disease':                     'autoimmune',
    'Parkinson disease':                        'autoimmune',
    'Toxic epidermal necrolysis':               'autoimmune',
    'Multiple sclerosis (MS)':                  'autoimmune',
    'Calcified Aortic Stenosis disease':        'autoimmune',
    'Diabetes Type 1':                          'autoimmune',
    'Rheumatoid arthritis':                     'autoimmune',
    'Celiac disease':                           'autoimmune',
    'Systemic lupus erythematosus':             'autoimmune',
    'Ankylosing spondylitis':                   'autoimmune',
    'Psoriasis':                                'autoimmune',
    'Inflammatory bowel disease':               'autoimmune',
    'Crohn\'s disease':                         'autoimmune',
    'Ulcerative colitis':                       'autoimmune',
    'Uveitis':                                  'autoimmune',
    'Vitiligo':                                 'autoimmune',
    'Alopecia areata':                          'autoimmune',
    'Graves disease':                           'autoimmune',
    'Hashimoto thyroiditis':                    'autoimmune',
}

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def clean_sequence(seq):
    if not isinstance(seq, str):
        return None
    seq = seq.strip().upper()
    if len(seq) < 8 or len(seq) > 30:
        return None
    if not all(c in VALID_AA for c in seq):
        return None
    return seq

def map_pathology(pathology):
    if not isinstance(pathology, str):
        return None
    p = pathology.strip()
    if p in LABEL_MAP:
        return LABEL_MAP[p]
    # fuzzy fallbacks for common variants
    pl = p.lower()
    if 'melanoma' in pl:        return 'cancer'
    if 'cancer' in pl:          return 'cancer'
    if 'carcinoma' in pl:       return 'cancer'
    if 'lymphoma' in pl:        return 'cancer'
    if 'leukemia' in pl:        return 'cancer'
    if 'tumor' in pl:           return 'cancer'
    if 'glioma' in pl:          return 'cancer'
    if 'sarcoma' in pl:         return 'cancer'
    if 'tuberculosis' in pl or 'tubercul' in pl: return 'bacterial'
    if 'influenza' in pl:       return 'viral'
    if 'cytomegalovirus' in pl or 'cmv' in pl:   return 'viral'
    if 'epstein' in pl or 'ebv' in pl:           return 'viral'
    if 'hiv' in pl or 'immunodeficiency' in pl:  return 'viral'
    if 'hepatitis' in pl:       return 'viral'
    if 'covid' in pl or 'sars' in pl:            return 'viral'
    if 'virus' in pl:           return 'viral'
    if 'herpes' in pl:          return 'viral'
    if 'diabetes' in pl:        return 'autoimmune'
    if 'sclerosis' in pl:       return 'autoimmune'
    if 'arthritis' in pl:       return 'autoimmune'
    if 'autoimmune' in pl:      return 'autoimmune'
    if 'allerg' in pl:          return 'autoimmune'
    if 'parkinson' in pl:       return 'autoimmune'
    if 'alzheimer' in pl:       return 'autoimmune'
    return None  # drop ambiguous

def clean_gene(gene):
    if not isinstance(gene, str):
        return None
    gene = gene.strip()
    if gene.lower() in ('', 'unknown', 'na', 'nan', 'none', '-'):
        return None
    # normalise colon notation → dash (TRBV25:01 → TRBV25-01)
    gene = gene.replace(':', '-')
    # remove leading zeros in allele suffix (TRBV6-01 → TRBV6-1)
    gene = re.sub(r'-0+(\d)', r'-\1', gene)
    return gene

def load_and_clean():
    df = pd.read_csv('data/TCR-Processed-Raw.csv')

    # rename columns to clean names
    df = df.rename(columns={
        'CDR3.beta.aa': 'cdr3',
        'Pathology':    'pathology',
        'TRBV':         'trbv',
        'TRBJ':         'trbj',
    })

    df['cdr3']      = df['cdr3'].apply(clean_sequence)
    df['label']     = df['pathology'].apply(map_pathology)
    df['trbv']      = df['trbv'].apply(clean_gene)
    df['trbj']      = df['trbj'].apply(clean_gene)

    # drop rows we can't use
    df = df.dropna(subset=['cdr3', 'label'])

    # deduplicate: same CDR3 + same label → keep one
    df = df.drop_duplicates(subset=['cdr3', 'label'])

    # if same CDR3 has conflicting labels, drop it entirely
    conflict = df.groupby('cdr3')['label'].nunique()
    conflict_seqs = conflict[conflict > 1].index
    df = df[~df['cdr3'].isin(conflict_seqs)]

    df = df[['cdr3', 'trbv', 'trbj', 'label']].reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = load_and_clean()
    print('Clean dataset shape:', df.shape)
    print('\nLabel distribution:')
    print(df['label'].value_counts())
    print('\nMissing TRBV:', df['trbv'].isna().sum())
    print('Missing TRBJ:', df['trbj'].isna().sum())
    print('\nSample:')
    print(df.head(5).to_string())
    df.to_csv('data/train_clean.csv', index=False)
    print('\nSaved to data/train_clean.csv')