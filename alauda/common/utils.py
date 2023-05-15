def get_as_pair(l):
    p1, p2 = l.split('\t')
    return p1, p2


# to customize for tf text standardization
def standardize_wp(in_str):
    return in_str
