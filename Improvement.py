#replace recombination function code withh given recombination function for improvement

def recombination(pop, moea):

    def get_genres(item):
        try:
            return set(moea.Si['genre'][item].split(moea.si_spilt))
        except:
            return set()

    pop_list = list(range(pop.shape[0]))

    while len(pop_list) != 0:

        tmp1 = random.choice(pop_list)
        pop_list.remove(tmp1)

        tmp2 = random.choice(pop_list)
        pop_list.remove(tmp2)

        y1 = pop[tmp1].copy()
        y2 = pop[tmp2].copy()

        if np.random.rand() < 0.9:

            u = random.randint(0, 1)
            v = random.randint(3, 4)

            tmp = y1[u:v].copy()
            y1[u:v] = y2[u:v]
            y2[u:v] = tmp

            for child in [y1, y2]:

                if len(set(child)) != moea.n_rec_movie:

                    current_genres = set()
                    for item in child:
                        current_genres |= get_genres(item)

                    duplicates = [x for x, count in Counter(child).items() if count > 1]

                    for dup in duplicates:

                        indices = indexA(child, dup)

                        for idx in indices[1:]:

                            candidates = list(set(moea.candidate).difference(child))

                            if len(candidates) == 0:
                                continue

                            best_item = None
                            best_score = -1

                            for c in candidates:

                                genres = get_genres(c)

                                new_genre = len(genres - current_genres) > 0
                                semantic_bonus = 0.5 if new_genre else 0

                                if semantic_bonus > best_score:
                                    best_score = semantic_bonus
                                    best_item = c

                            child[idx] = best_item
                            current_genres |= get_genres(best_item)

        pop[tmp1] = y1
        pop[tmp2] = y2

    return pop

