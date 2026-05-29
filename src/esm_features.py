import torch
import esm
import numpy as np
import pandas as pd
import pickle
import os

def load_esm_model():
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

def get_esm_embeddings(sequences, ids, model, alphabet, batch_converter,
                       batch_size=32):
    all_embeddings = {}
    for i in range(0, len(sequences), batch_size):
        batch_ids  = ids[i:i+batch_size]
        batch_seqs = sequences[i:i+batch_size]
        data = [(pid, seq.upper()) for pid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12], return_contacts=False)
        token_embeddings = results['representations'][12]
        for j, pid in enumerate(batch_ids):
            seq_len = len(batch_seqs[j])
            # mean-pool over sequence positions (exclude BOS/EOS tokens)
            emb = token_embeddings[j, 1:seq_len+1].mean(0).numpy()
            all_embeddings[pid] = emb
        if (i // batch_size) % 5 == 0:
            print(f'  Processed {min(i+batch_size, len(sequences))}/{len(sequences)}')
    return all_embeddings

def generate_and_save_embeddings(csv_path, out_path, id_col=None):
    df = pd.read_csv(csv_path)

    if id_col and id_col in df.columns:
        ids = df[id_col].tolist()
    else:
        ids = [f'seq_{i}' for i in range(len(df))]

    seq_col = 'cdr3' if 'cdr3' in df.columns else 'CDR3.beta.aa'
    sequences = df[seq_col].tolist()

    print('Loading ESM-2 35M model...')
    model, alphabet, batch_converter = load_esm_model()
    print('Model loaded. Generating embeddings...')

    embeddings = get_esm_embeddings(sequences, ids, model, alphabet,
                                    batch_converter)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f'Saved {len(embeddings)} embeddings to {out_path}')
    return embeddings

if __name__ == '__main__':
    generate_and_save_embeddings(
        csv_path='data/train_clean.csv',
        out_path='models/esm_embeddings_train.pkl',
        id_col=None
    )
    generate_and_save_embeddings(
        csv_path='data/test_set.csv',
        out_path='models/esm_embeddings_test.pkl',
        id_col='ID'
    )