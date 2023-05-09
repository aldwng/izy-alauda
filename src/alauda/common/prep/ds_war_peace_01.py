import random
import re

corpora_in = 'C:/Dev/ds/war_peace_plain_in.txt'
prep_out = 'C:/Dev/ds/wp_prep_01.txt'
prep_train = 'C:/Dev/ds/wp_prep_01_train.txt'
prep_valid = 'C:/Dev/ds/wp_prep_01_valid.txt'
prep_test = 'C:/Dev/ds/wp_prep_01_test.txt'

kick_terms = {'a', 'the', 'i', 'he', 'she', 'him', 'her', 'his', 'and', 'or', 'an', 'is', 'was', 'am', 'were', 'are',
              'you'}

window_size_min = 2
window_size_max = 5
window_size_fix = 6


def txt_filter(s):
    return (not s.startswith('CHAPTER')) and (s != '') and (s is not None)


def clean_sentence(s):
    ns = s.replace('’', '')
    ns = re.sub(r'[^A-Za-z ]+', ' ', ns)
    ns = re.sub('\s+', ' ', ns)
    ns = ns.strip()
    ns = ns.casefold()
    return ns


def sample(sentence, w_size):
    terms = sentence.split()
    if len(terms) <= w_size + 1:
        return ''
    w_terms = terms[0: w_size]
    window = ' '.join(w_terms)

    return window + '\t' + ' '.join(terms[w_size:])


with open(corpora_in, encoding='UTF-8') as f:
    lines = f.read().split('\n')

lines = list(filter(txt_filter, lines))

one_line = ' '.join(lines)

lines = one_line.split(".")

lines = list(filter(None, lines))

for i in range(len(lines)):
    lines[i] = clean_sentence(lines[i])

seqs = []
for line in lines:
    seq = sample(line, window_size_fix)
    if seq is not None and seq != '':
        seqs.append(seq)

print(seqs)

out_file = open(prep_out, 'w')
for line in seqs:
    out_file.write(line + '\n')
out_file.close()

seq_pairs = []
for seq in seqs:
    seq1, seq2 = seq.split('\t')
    seq2 = '[SOS] ' + seq2 + ' [EOS]'
    seq_pairs.append((seq1, seq2))

random.shuffle(seq_pairs)
num_valid = int(0.15 * len(seq_pairs))
num_train = len(seq_pairs) - 2 * num_valid
seq_pairs_train = seq_pairs[:num_train]
seq_pairs_valid = seq_pairs[num_train:num_train + num_valid]
seq_pairs_test = seq_pairs[num_train + num_valid:]

out_train = open(prep_train, 'w')
for pair in seq_pairs_train:
    out_train.write(pair[0] + '\t' + pair[1] + '\n')
out_train.close()

out_valid = open(prep_valid, 'w')
for pair in seq_pairs_valid:
    out_valid.write(pair[0] + '\t' + pair[1] + '\n')
out_valid.close()

out_test = open(prep_test, 'w')
for pair in seq_pairs_test:
    out_test.write(pair[0] + '\t' + pair[1] + '\n')
out_test.close()

print(len(seqs))
