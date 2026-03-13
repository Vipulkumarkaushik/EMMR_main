import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking

# ===============================
# CONFIG
# ===============================
dataset = "ml-10m"

train_file = "Code/data/" + dataset + ".train"
output_file = "Code/util_data/BPRMF" + dataset + "_rating.npy"

TOP_N = 100
FACTORS = 64
ITERATIONS = 100

# ===============================
# LOAD DATA
# ===============================
print("Loading dataset...")

data = pd.read_csv(train_file, sep=",", header=None)
data.columns = ["user", "item"]

n_users = data["user"].max() + 1
n_items = data["item"].max() + 1

print("Users:", n_users)
print("Items:", n_items)

# ===============================
# BUILD MATRIX
# ===============================
print("Building sparse matrix...")

user_item_matrix = sp.coo_matrix(
    (np.ones(len(data)), (data["user"], data["item"])),
    shape=(n_users, n_items)
).tocsr()

# implicit needs item-user matrix
item_user_matrix = user_item_matrix.T.tocsr()

# ===============================
# TRAIN BPR
# ===============================
print("Training BPR-MF...")

model = BayesianPersonalizedRanking(
    factors=FACTORS,
    learning_rate=0.01,
    iterations=ITERATIONS
)

model.fit(item_user_matrix)

print("Training complete.")

# ===============================
# GENERATE CANDIDATES
# ===============================
print("Generating candidate items...")

user_factors = model.item_factors      # corresponds to real users
item_factors = model.user_factors      # corresponds to items

rating = {}

for u in range(n_users):

    if u % 500 == 0:
        print("Processing user:", u)

    scores = user_factors[u].dot(item_factors.T)

    # remove already interacted items
    interacted = user_item_matrix[u].indices
    scores[interacted] = -np.inf

    # top 100
    top_items = np.argpartition(scores, -TOP_N)[-TOP_N:]
    top_items = top_items[np.argsort(scores[top_items])[::-1]]

    rating[u] = {}

    for item in top_items:
        rating[u][int(item)] = float(scores[item])

# ===============================
# SAVE FILE
# ===============================
print("Saving .npy file...")

np.save(output_file, rating)

print("Done.")
print("Saved at:", output_file)