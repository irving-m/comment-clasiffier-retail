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

words1 = Counter(txtlist)
words2 = Counter(txtlist2)
words3 = Counter(txtlist3)

gram1 = dict(words1.most_common())
gram2 = dict(words2.most_common())
gram3 = dict(words3.most_common())


gram1 = {k: v for k, v in gram1.items() if v != 1}
gram2 = {k: v for k, v in gram2.items() if v != 1}
gram3 = {k: v for k, v in gram3.items() if v != 1}   

n1 = sum(gram1.values())
n2 = sum(gram2.values())
n3 = sum(gram3.values())

gram1prob = {k: v/n1 for k, v in gram1.items()}
gram2prob = {k: v/n2 for k, v in gram2.items()}
gram3prob = {k: v/n3 for k, v in gram3.items()}





with open(os.path.join(path2, "gram1prob.pickle"), "wb") as f:
    pickle.dump(gram1prob, f)

with open(os.path.join(path2, "gram2prob.pickle"), "wb") as f:
    pickle.dump(gram2prob, f)

with open(os.path.join(path2, "gram3prob.pickle"), "wb") as f:
    pickle.dump(gram3prob, f)




