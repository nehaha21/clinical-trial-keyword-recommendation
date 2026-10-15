"""Live demo: hide half of a test trial's MeSH terms and let the trained model recommend the missing ones.

Run from the project root:   python -m scripts.demo
(save this file as scripts/demo.py). Press Enter for another trial, type q to quit.
It does NOT train anything; it only loads the saved model.
"""
import pickle
import numpy as np
from ml.data_set_splitter import DataSetSplitter

SDS = './output/ctg-public-xml.sds'
MODEL = './output/model-0.001_1000.0_0.01_0.1_60.pickle'
CTDMC = './output/ctg-public-xml.ctdmc'

print("Loading trained model...")
with open(MODEL, 'rb') as f:
    model = pickle.load(f)

print("Loading MeSH term names...")
with open(CTDMC, 'rb') as f:
    ctdmc = pickle.load(f)
names = ctdmc._mesh_trie._mesh_indexer            # mesh_index -> term name
mesh_index_map = DataSetSplitter.get_mesh_index_map(SDS)   # matrix row -> mesh_index


def name(row):
    return names[int(mesh_index_map[int(row)])]


def show(rows, limit=12):
    rows = list(rows)
    txt = ", ".join(name(r) for r in rows[:limit])
    return txt + (f", ... (+{len(rows) - limit} more)" if len(rows) > limit else "")


print("Loading test trials (a few minutes, only once)...")
TEST = DataSetSplitter.get_test_utility_matrix(SDS)
n_docs = TEST.shape[1]
rng = np.random.default_rng()
print(f"Ready: {n_docs} test trials.\n")

while True:
    j = int(rng.integers(n_docs))
    col = TEST[:, j].toarray().ravel()
    present = np.flatnonzero(col)
    if len(present) < 6:
        continue
    hidden = rng.choice(present, size=len(present) // 2, replace=False)
    shown = np.setdiff1d(present, hidden)
    masked = col.copy()
    masked[hidden] = 0

    pred = model.predict(masked.reshape(-1, 1)).ravel().astype(np.float64)
    pred[shown] = -1.0                       # do not re-recommend terms the model was already given
    top3 = np.argsort(pred)[::-1][:3]

    print("=" * 70)
    print(f"Test trial #{j}: {len(present)} MeSH terms, half hidden from the model")
    print("Terms shown to the model :", show(shown))
    print("Hidden terms (answer)    :", show(hidden))
    print("Model's top-3 suggestions:")
    hits = 0
    for r in top3:
        ok = col[r] > 0
        hits += int(ok)
        print(f"   {'YES' if ok else 'no '}  {name(r)}")
    print(f"Correct: {hits} of 3")
    if input("\nEnter = another trial, q = quit: ").strip().lower() == 'q':
        break