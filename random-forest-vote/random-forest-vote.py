def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    trees_final = list()
    for tree_votes in zip(*predictions):
        votes = {k: 0 for k in set(tree_votes)}
        for sample in tree_votes:
            votes[sample] += 1
        largest_vote = "-1"
        for k,v in votes.items():
            if largest_vote == "-1":
                largest_vote = k
            elif v > votes[largest_vote]:
                largest_vote = k
            else:
                continue
        trees_final.append(int(largest_vote))
    return trees_final