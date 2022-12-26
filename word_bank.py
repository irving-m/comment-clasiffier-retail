# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:32:48 2022

@author: Irving
"""
import pickle
import os
from collections import Counter
import re

regex = re.compile(r"[^a-zñáéíóúü\s]")

path = os.path.join(os.getcwd(), "Libros")

path2 = os.path.join(os.getcwd(), "Big lists")

with open(os.path.join(path2, "bigtexto.txt"), "r", encoding = "utf-8") as r:
    txt = " ".join(r.readlines())
    
txt2 = regex.sub(" ", txt.lower())
txtlist = txt2.split()
txtlist2 = [" ".join(txtlist[i:i + 2]) for i, w in enumerate(txtlist)]
txtlist3 = [" ".join(txtlist[i:i + 3]) for i, w in enumerate(txtlist)]

def get_prob(text_list):
    words = Counter(text_list)
    gram = dict(words.most_common())
    gram = {k: v for k, v in gram.items() if v != 1}
    n = sum(gram.values())

    return {k: v/n for k, v in gram.items()}
    

gram1prob = get_prob(txtlist)
gram2prob = get_prob(txtlist2)
gram3prob = get_prob(txtlist3)


with open(os.path.join(path2, "gram1prob.pickle"), "wb") as f:
    pickle.dump(gram1prob, f)

with open(os.path.join(path2, "gram2prob.pickle"), "wb") as f:
    pickle.dump(gram2prob, f)

with open(os.path.join(path2, "gram3prob.pickle"), "wb") as f:
    pickle.dump(gram3prob, f)


