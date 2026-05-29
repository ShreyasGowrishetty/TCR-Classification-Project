import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import extract_features

LABEL_ORDER = ['viral', 'bacterial', 'cancer', 'autoimmune']
LABEL2IDX   = {l: i for i, l in enumerate(LABEL_ORDER)}
# viral=0, bacterial=1, cancer=2, autoimmune=3

def load_esm_embeddings(path, df):
    with open(path, 'rb') as f:
        emb_dict = pickle.load(f)
    ids = [f'seq_{i}' for i in range(len(df))]
    dim = next(iter(emb_dict.values())).shape[0]
    matrix = np.zeros((len(df), dim), dtype=np.float32)
    for i, sid in enumerate(ids):
        if sid in emb_dict:
            matrix[i] = emb_dict[sid]
    return matrix

def make_xgb():
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=1,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
    )

def compute_sample_weights(y, power=1.5):
    class_counts  = np.bincount(y, minlength=4)
    class_weights = (len(y) / (4 * class_counts)) ** power
    return class_weights[y]

def train_and_save(use_genes=True):
    suffix = 'gene' if use_genes else 'seq'
    print(f'\n{"="*50}')
    print(f'Training mode: {"seq+gene" if use_genes else "seq-only"}')
    print(f'{"="*50}')

    df = pd.read_csv('data/train_clean.csv')

    # correct label encoding: viral=0, bacterial=1, cancer=2, autoimmune=3
    y = np.array([LABEL2IDX[l] for l in df['label']])
    print('Label distribution (encoded):')
    for label, idx in LABEL2IDX.items():
        print(f'  {idx} -> {label}: {(y == idx).sum()}')

    X_hand, trbv_vocab, trbj_vocab = extract_features(df, use_genes=use_genes)
    X_esm  = load_esm_embeddings('models/esm_embeddings_train.pkl', df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(df), 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_esm, y)):
        print(f'\n  Fold {fold+1}/5')

        esm_scaler  = StandardScaler()
        hand_scaler = StandardScaler()

        X_esm_tr  = esm_scaler.fit_transform(X_esm[train_idx])
        X_esm_val = esm_scaler.transform(X_esm[val_idx])
        X_hand_tr  = hand_scaler.fit_transform(X_hand[train_idx])
        X_hand_val = hand_scaler.transform(X_hand[val_idx])

        X_tr  = np.hstack([X_esm_tr,  X_hand_tr])
        X_val = np.hstack([X_esm_val, X_hand_val])
        y_tr, y_val = y[train_idx], y[val_idx]

        sample_weights = compute_sample_weights(y_tr, power=1.5)

        clf = make_xgb()
        clf.fit(X_tr, y_tr, sample_weight=sample_weights,
                eval_set=[(X_val, y_val)], verbose=False)

        oof_preds[val_idx] = clf.predict_proba(X_val)
        fold_f1 = f1_score(y_val, oof_preds[val_idx].argmax(axis=1),
                           average='macro')
        print(f'    Macro F1: {fold_f1:.4f}')

    oof_labels = oof_preds.argmax(axis=1)
    macro_f1   = f1_score(y, oof_labels, average='macro')
    print(f'\nOOF Macro F1: {macro_f1:.4f}')
    print('\nPer-class report:')
    print(classification_report(y, oof_labels, target_names=LABEL_ORDER))

    print('Retraining on full dataset...')
    esm_scaler_full  = StandardScaler()
    hand_scaler_full = StandardScaler()
    X_esm_full  = esm_scaler_full.fit_transform(X_esm)
    X_hand_full = hand_scaler_full.fit_transform(X_hand)
    X_full      = np.hstack([X_esm_full, X_hand_full])

    sample_weights = compute_sample_weights(y, power=1.5)
    clf_full = make_xgb()
    clf_full.fit(X_full, y, sample_weight=sample_weights, verbose=False)

    os.makedirs('models', exist_ok=True)
    with open(f'models/model_{suffix}.pkl',       'wb') as f: pickle.dump(clf_full,        f)
    with open(f'models/esm_scaler_{suffix}.pkl',  'wb') as f: pickle.dump(esm_scaler_full,  f)
    with open(f'models/hand_scaler_{suffix}.pkl', 'wb') as f: pickle.dump(hand_scaler_full, f)
    with open(f'models/label_encoder.pkl',        'wb') as f: pickle.dump(LABEL2IDX,        f)
    with open(f'models/trbv_vocab_{suffix}.pkl',  'wb') as f: pickle.dump(trbv_vocab,       f)
    with open(f'models/trbj_vocab_{suffix}.pkl',  'wb') as f: pickle.dump(trbj_vocab,       f)

    print(f'Saved models/model_{suffix}.pkl')
    return macro_f1

if __name__ == '__main__':
    f1_seq  = train_and_save(use_genes=False)
    f1_gene = train_and_save(use_genes=True)
    print(f'\n{"="*50}')
    print(f'SUMMARY')
    print(f'{"="*50}')
    print(f'Seq-only  OOF Macro F1: {f1_seq:.4f}')
    print(f'Seq+gene  OOF Macro F1: {f1_gene:.4f}')