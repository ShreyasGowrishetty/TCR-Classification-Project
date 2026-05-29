import numpy as np
import pandas as pd
import pickle
from features import extract_features

LABEL_ORDER = ['viral', 'bacterial', 'cancer', 'autoimmune']

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_esm_embeddings(path, ids):
    emb_dict = load_pickle(path)
    dim = next(iter(emb_dict.values())).shape[0]
    matrix = np.zeros((len(ids), dim), dtype=np.float32)
    for i, sid in enumerate(ids):
        if sid in emb_dict:
            matrix[i] = emb_dict[sid]
    return matrix

def predict(use_genes=True):
    suffix = 'gene' if use_genes else 'seq'
    print(f'\nGenerating predictions: {"seq+gene" if use_genes else "seq-only"} mode')

    df = pd.read_csv('data/test_set.csv')
    ids = df['ID'].tolist()

    # rename columns to match what features.py expects
    df = df.rename(columns={
        'CDR3.beta.aa': 'cdr3',
        'TRBV':        'trbv',
        'TRBJ':        'trbj',
    })

    # load vocabs built during training
    trbv_vocab = load_pickle(f'models/trbv_vocab_{suffix}.pkl')
    trbj_vocab = load_pickle(f'models/trbj_vocab_{suffix}.pkl')

    # extract handcrafted features using training vocabs
    X_hand, _, _ = extract_features(df, trbv_vocab=trbv_vocab,
                                    trbj_vocab=trbj_vocab,
                                    use_genes=use_genes)

    # load ESM embeddings (keyed by ID for test set)
    emb_dict    = load_pickle('models/esm_embeddings_test.pkl')
    dim         = next(iter(emb_dict.values())).shape[0]
    X_esm       = np.zeros((len(ids), dim), dtype=np.float32)
    for i, sid in enumerate(ids):
        if sid in emb_dict:
            X_esm[i] = emb_dict[sid]

    # scale using training scalers
    esm_scaler  = load_pickle(f'models/esm_scaler_{suffix}.pkl')
    hand_scaler = load_pickle(f'models/hand_scaler_{suffix}.pkl')
    X_esm_s     = esm_scaler.transform(X_esm)
    X_hand_s    = hand_scaler.transform(X_hand)
    X           = np.hstack([X_esm_s, X_hand_s])

    # predict
    model  = load_pickle(f'models/model_{suffix}.pkl')
    probas = model.predict_proba(X)
    preds  = [LABEL_ORDER[i] for i in probas.argmax(axis=1)]

    # build submission
    sub = pd.DataFrame({'ID': ids, 'prediction': preds})
    out_path = f'submission_{suffix}.csv'
    sub.to_csv(out_path, index=False)
    print(f'Saved {out_path}')
    print('\nPrediction distribution:')
    print(sub['prediction'].value_counts())
    print('\nFirst 5 rows:')
    print(sub.head().to_string())

    return sub

if __name__ == '__main__':
    predict(use_genes=False)
    predict(use_genes=True)