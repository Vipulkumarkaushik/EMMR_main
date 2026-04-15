import numpy as np
import random
from collections import Counter

def indexA(L, val):
    return [i for i, v in enumerate(L) if v == val]

def recombination(pop, moea):
    alpha = 0.7
    beta = 0.3
    lam = 1.0
    
    pop_list = list(range(pop.shape[0]))

    def compute_contextual_embedding(items):
        embeddings = [moea.item_embeddings[item] for item in items]
        return np.mean(embeddings, axis=0)

    def cosine_similarity(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(v1, v2) / (norm1 * norm2)

    while len(pop_list) >= 2:
        tmp1 = random.choice(pop_list)
        pop_list.remove(tmp1)
        tmp2 = random.choice(pop_list)
        pop_list.remove(tmp2)

        y1 = pop[tmp1].copy()
        y2 = pop[tmp2].copy()

        if np.random.rand() < 0.9:
            u = random.randint(0, len(y1) - 2)
            v = random.randint(u + 1, len(y1))
            y1[u:v], y2[u:v] = y2[u:v].copy(), y1[u:v].copy()

        for child_idx, child in enumerate([y1, y2]):
            user_idx = tmp1 if child_idx == 0 else tmp2
            
            if len(set(child)) != moea.n_rec_movie:
                psi_x = compute_contextual_embedding(child)
                counts = Counter(child)
                duplicates = [item for item, count in counts.items() if count > 1]
                
                for dup in duplicates:
                    indices = indexA(child.tolist(), dup)
                    for idx in indices[1:]:
                        all_genres = list(moea.genre_embeddings.keys())
                        probs = []
                        for g_name in all_genres:
                            g_emb = moea.genre_embeddings[g_name]
                            alignment = cosine_similarity(psi_x, g_emb)
                            probs.append(np.exp(-lam * alignment))
                        
                        probs = np.array(probs) / np.sum(probs)
                        genre_star = np.random.choice(all_genres, p=probs)
                        
                        candidates = list(set(moea.genre_items[genre_star]).difference(set(child)))
                        if not candidates:
                            candidates = list(set(moea.candidate).difference(set(child)))
                        
                        best_item = None
                        max_score = -float('inf')
                        
                        for c in candidates:
                            e_c = moea.item_embeddings[c]
                            sim_c_x = cosine_similarity(e_c, psi_x)
                            r_uc = moea.predicted_ratings[user_idx][c]
                            score = alpha * r_uc + beta * (1 - sim_c_x)
                            
                            if score > max_score:
                                max_score = score
                                best_item = c
                        
                        if best_item is not None:
                            child[idx] = best_item
                            psi_x = compute_contextual_embedding(child)

        pop[tmp1] = y1
        pop[tmp2] = y2

    return pop
