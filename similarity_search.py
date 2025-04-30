import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Process, Queue
import contextlib
import argparse

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def _mcs_worker(mol1_smiles, mol2_smiles, queue):
    try:
        mol1 = Chem.MolFromSmiles(mol1_smiles)
        mol2 = Chem.MolFromSmiles(mol2_smiles)
        mcs = rdFMCS.FindMCS([mol1, mol2], completeRingsOnly=True)
        if mcs.smartsString == "":
            queue.put(0.0)
        else:
            common = Chem.MolFromSmarts(mcs.smartsString)
            percent = (common.GetNumAtoms() / mol1.GetNumAtoms()) * 100
            queue.put(percent)
    except Exception:
        queue.put(0.0)

def safe_mcs(mol1, mol2, timeout=2):
    queue = Queue()
    proc = Process(target=_mcs_worker, args=(Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2), queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0.0
    try:
        return queue.get_nowait()
    except:
        return 0.0

def compute_similarity(spec1, mzs, intensities, precursor_mz):
    spec2 = Spectrum(mz=mzs, intensities=intensities, metadata={"precursor_mz": precursor_mz})
    score = ModifiedCosine(tolerance=0.01).pair(spec1, spec2)
    return float(score["score"])

def process_row_optimized(row_idx, mzs, intensities, smiles, precursor_mzs, inchikeys, mass_window=20, top_k=10):
    with suppress_stderr():
        inchikey_q = inchikeys[row_idx]
        precursor_q = precursor_mzs[row_idx]
        mol_q = Chem.MolFromSmiles(smiles[row_idx])
        spec_q = Spectrum(mz=mzs[row_idx], intensities=intensities[row_idx], metadata={"precursor_mz": precursor_q})

        candidates = [i for i in range(len(inchikeys)) if i != row_idx and abs(precursor_mzs[i] - precursor_q) <= mass_window]
        sim_scores = [(i, compute_similarity(spec_q, mzs[i], intensities[i], precursor_mzs[i])) for i in candidates]
        top_candidates = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_k]

        best_score = -1.0
        best_mcs = -1.0
        best_combined = -1.0
        best_inchikey = None

        for i, score in top_candidates:
            mol_c = Chem.MolFromSmiles(smiles[i])
            mcs = safe_mcs(mol_q, mol_c, timeout=2)
            if mcs == 0.0:
                continue
            combined = min(mcs / 100, score)
            if combined > best_combined:
                best_combined = combined
                best_score = score
                best_mcs = mcs
                best_inchikey = inchikeys[i]

        return {
            "inchikey_query": inchikey_q,
            "inchikey_similar": best_inchikey,
            "matchms_similarity": best_score,
            "mcs_percent": best_mcs,
            "combined_score": best_combined
        }

def full_similarity_search_fast(df, n_jobs=4, mass_window=20, top_k=10, batch_size=1000):
    mzs = df["mzs"].tolist()
    intensities = df["intensities"].tolist()
    smiles = df["smiles"].tolist()
    precursor_mzs = df["precursor_mz"].tolist()
    inchikeys = df["inchikey"].tolist()

    total = len(df)
    results_all = []

    with parallel_backend('loky'):
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_indices = list(range(start_idx, end_idx))
            print(f"\n🚀 Processing {start_idx}–{end_idx}...")

            batch_results = []
            for count, i in enumerate(batch_indices, start=1):
                result = process_row_optimized(i, mzs, intensities, smiles, precursor_mzs, inchikeys, mass_window, top_k)
                batch_results.append(result)

                if count % 100 == 0 or count == len(batch_indices):
                    print(f"✅ Processed {start_idx + count} rows")

            df_batch = pd.DataFrame(batch_results)
            results_all.append(df_batch)
            df_batch.to_csv(f"similarity_results_part_{start_idx}_{end_idx}.csv", index=False)
            print(f"💾 Saved batch {start_idx}-{end_idx}")

    final_df = pd.concat(results_all, ignore_index=True)
    print("\n🎉 All done!")
    return final_df


# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity search")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()

    input_path = args.input
    print(f"📂 Loading data from: {input_path}")

    df = pd.read_csv(input_path)
    df["mzs"] = df["mzs"].apply(lambda x: np.array(list(map(float, x.strip("[]").split()))))
    df["intensities"] = df["intensities"].apply(lambda x: np.array(list(map(float, x.strip("[]").split()))))

    results = full_similarity_search_fast(df, n_jobs=8)
    results.to_csv("similarity_results_all.csv", index=False)