import random
import numpy as np


def generate_rand_haplotype(hap_len):
    hap = ""
    for i in range(hap_len):
        hap += str(random.randint(0, 1))
    return hap


def get_complement_char(c):
    if c == "0":
        return "1"
    else:
        return "0"


def get_complement_haplotype(hap):
    hap_c = ""
    for i in range(len(hap)):
        hap_c += get_complement_char(hap[i])
    return hap_c


def get_random_substring(s, l):
    assert(l <= len(s))
    start = random.randint(0, len(s) - l)
    return s[start:start + l], start


class HaplotypeFrag:
    def __init__(self, seq, start):
        self.start = start
        self.seq = seq
        self.end = start + len(seq) - 1

    def len(self):
        return len(self.seq)

    def overlaps(self, f):
        return min(self.end, f.end) - max(self.start, f.start)


def get_n_random_substrings_normal_dist(s, n, err=0, mu=5, sig=0.1):
    strings = []
    for i in range(n):
        l = np.random.normal(mu, sig, 1).round().astype(int)[0]
        ss, start = get_random_substring(s, l)
        ss_err = ""
        for j in range(len(ss)):
            flip = random.random() < err
            if flip:
                ss_err += get_complement_char(ss[j])
            else:
                ss_err += ss[j]
        strings.append(HaplotypeFrag(ss_err, start))
    return strings

