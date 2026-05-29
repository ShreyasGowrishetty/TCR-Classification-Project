import pandas as pd
import numpy as np
from itertools import product

# ── Amino acid property tables ─────────────────────────────────────────────
KD_HYDROPHOBICITY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G':-0.4, 'T':-0.7, 'S':-0.8,
    'W':-0.9, 'Y':-1.3, 'P':-1.6, 'H':-3.2, 'E':-3.5,
    'Q':-3.5, 'D':-3.5, 'N':-3.5, 'K':-3.9, 'R':-4.5,
}
CHARGE = {
    'R': 1, 'K': 1, 'H': 0.1,
    'D':-1, 'E':-1,
}
AROMATIC  = set('FWY')
POLAR     = set('STNQ')
NONPOLAR  = set('AVILMFYWP')
CHARGED   = set('RKDE')

ALL_AA = list('ACDEFGHIKLMNPQRSTVWY')
ALL_2MER = [''.join(p) for p in product(ALL_AA, repeat=2)]

# ── Helpers ────────────────────────────────────────────────────────────────
def safe_mean(vals):
    return float(np.mean(vals)) if vals else 0.0

def aa_props(seq):
    n = len(seq)
    hydro   = safe_mean([KD_HYDROPHOBICITY.get(a, 0) for a in seq])
    charge  = sum(CHARGE.get(a, 0) for a in seq)
    aromatic= sum(1 for a in seq if a in AROMATIC) / n
    polar   = sum(1 for a in seq if a in POLAR)    / n
    nonpolar= sum(1 for a in seq if a in NONPOLAR) / n
    charged = sum(1 for a in seq if a in CHARGED)  / n
    return hydro, charge, aromatic, polar, nonpolar, charged

def kmer_counts(seq, k=2):
    counts = {km: 0 for km in ALL_2MER}
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k]
        if km in counts:
            counts[km] += 1
    total = max(len(seq) - k + 1, 1)
    return [counts[km] / total for km in ALL_2MER]

def positional_features(seq):
    """One-hot encode first 3, last 2, and middle 3 amino acids."""
    feats = []
    # N-terminal: positions 0,1,2
    for i in range(3):
        aa = seq[i] if i < len(seq) else 'X'
        feats += [1 if aa == a else 0 for a in ALL_AA]
    # C-terminal: last 2
    for i in [-2, -1]:
        aa = seq[i] if len(seq) >= 2 else 'X'
        feats += [1 if aa == a else 0 for a in ALL_AA]
    # center 3 residues
    mid = len(seq) // 2
    for i in [mid-1, mid, mid+1]:
        aa = seq[i] if 0 <= i < len(seq) else 'X'
        feats += [1 if aa == a else 0 for a in ALL_AA]
    return feats  # 3+2+3 = 8 positions × 20 AA = 160 features

# ── V/J gene encoding ──────────────────────────────────────────────────────
def build_gene_vocabs(df):
    trbv_vals = df['trbv'].dropna().unique().tolist()
    trbj_vals = df['trbj'].dropna().unique().tolist()
    trbv_vocab = {g: i for i, g in enumerate(sorted(trbv_vals))}
    trbj_vocab = {g: i for i, g in enumerate(sorted(trbj_vals))}
    return trbv_vocab, trbj_vocab

def gene_features(trbv, trbj, trbv_vocab, trbj_vocab):
    feats = []
    # TRBV one-hot
    v_vec = [0] * len(trbv_vocab)
    v_missing = 1 if trbv is None or (isinstance(trbv, float) and np.isnan(trbv)) else 0
    if not v_missing and trbv in trbv_vocab:
        v_vec[trbv_vocab[trbv]] = 1
    feats += v_vec + [v_missing]
    # TRBJ one-hot
    j_vec = [0] * len(trbj_vocab)
    j_missing = 1 if trbj is None or (isinstance(trbj, float) and np.isnan(trbj)) else 0
    if not j_missing and trbj in trbj_vocab:
        j_vec[trbj_vocab[trbj]] = 1
    feats += j_vec + [j_missing]
    # V gene family (e.g. TRBV12-3 → family 12)
    v_family = 0
    if not v_missing and isinstance(trbv, str):
        import re
        m = re.match(r'TRBV(\d+)', trbv)
        v_family = int(m.group(1)) if m else 0
    feats += [v_family]
    return feats

# ── Main feature extractor ─────────────────────────────────────────────────
def extract_features(df, trbv_vocab=None, trbj_vocab=None, use_genes=True):
    """
    Returns X (np.ndarray) and feature names.
    If trbv_vocab/trbj_vocab are None, builds them from df (training mode).
    Pass them in for test-time extraction.
    """
    if trbv_vocab is None or trbj_vocab is None:
        trbv_vocab, trbj_vocab = build_gene_vocabs(df)

    rows = []
    for _, row in df.iterrows():
        seq  = row['cdr3']
        trbv = row.get('trbv', None)
        trbj = row.get('trbj', None)
        if isinstance(trbv, float) and np.isnan(trbv): trbv = None
        if isinstance(trbj, float) and np.isnan(trbj): trbj = None

        feats = []

        # 1. Length features
        n = len(seq)
        feats += [n, n**2, 1 if n <= 12 else 0, 1 if 13 <= n <= 16 else 0, 1 if n >= 17 else 0]

        # 2. Global physicochemical
        hydro, charge, aromatic, polar, nonpolar, charged = aa_props(seq)
        feats += [hydro, charge, aromatic, polar, nonpolar, charged]

        # 3. Loop-only physicochemical (central 60% of sequence)
        start = max(0, int(n * 0.2))
        end   = min(n, int(n * 0.8))
        loop  = seq[start:end] if end > start else seq
        lhydro, lcharge, laromatic, lpolar, lnonpolar, lcharged = aa_props(loop)
        feats += [lhydro, lcharge, laromatic, lpolar, lnonpolar, lcharged]

        # 4. Amino acid composition (global)
        comp = [seq.count(a) / n for a in ALL_AA]
        feats += comp

        # 5. Positional one-hot (N-term, C-term, center)
        feats += positional_features(seq)

        # 6. 2-mer frequencies
        feats += kmer_counts(seq, k=2)

        # 7. V/J gene features
        if use_genes:
            feats += gene_features(trbv, trbj, trbv_vocab, trbj_vocab)

        rows.append(feats)

    return np.array(rows, dtype=np.float32), trbv_vocab, trbj_vocab

# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('data/train_clean.csv')
    print(f'Loaded {len(df)} rows')

    print('Extracting seq+gene features...')
    X_gene, trbv_vocab, trbj_vocab = extract_features(df, use_genes=True)
    print(f'  Shape: {X_gene.shape}')

    print('Extracting seq-only features...')
    X_seq, _, _ = extract_features(df, trbv_vocab, trbj_vocab, use_genes=False)
    print(f'  Shape: {X_seq.shape}')

    print('\nNo NaNs in gene features:', not np.isnan(X_gene).any())
    print('No NaNs in seq features: ', not np.isnan(X_seq).any())
    print('\nSample feature vector (first 10 values):', X_gene[0, :10])