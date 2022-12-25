# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:21:14 2022

@author: Irving
"""

import pickle
import os

path = r"C:\Users\Irving\PyCharm Projects\Talking Tom\Big lists"


with open(os.path.join(path, "gram1prob.pickle"), "rb") as g:
    gram1 = pickle.load(g)

with open(os.path.join(path, "gram2prob.pickle"), "rb") as g:
    gram2 = pickle.load(g)

with open(os.path.join(path, "gram3prob.pickle"), "rb") as g:
    gram3 = pickle.load(g)
    

def detector(sentence_list): 
    return [i not in gram1 for i in sentence_list]


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'aábcdeéfghiíjklmnñoópqrstuúüvwxyz'
    
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word): 
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


def known(words): 
    return set(w for w in words if w in gram1)


def get_candidates(word):
    distance1 = known(edits1(word))
    if distance1:
        return distance1
    else:
        distance2 = known(edits2(word))
        if distance2:
            return distance2
        else:
            return 0
        
  
def replace_gram(L, U, sentence_list, pivot, candidate):
    string_ngram = " ".join(sentence_list[max(pivot + L, 0): pivot + U])
    
    return string_ngram.replace(sentence_list[pivot], candidate)
            

def ngram(sentence_list, candidates, pivot):
    ngrams = {}
    
    for candidate in candidates:
        
        ngrams[replace_gram(0, 3, sentence_list, pivot, candidate)] = candidate
        ngrams[replace_gram(-1, 2, sentence_list, pivot, candidate)] = candidate
        ngrams[replace_gram(-2, 1, sentence_list, pivot, candidate)] = candidate
        
        ngrams[replace_gram(0, 2, sentence_list, pivot, candidate)] = candidate
        ngrams[replace_gram(-1, 1, sentence_list, pivot, candidate)] = candidate
        
        ngrams[replace_gram(0, 1, sentence_list, pivot, candidate)] = candidate
    
    return ngrams


def search_dict(candidate, previous, c, dictionary):
    if candidate in dictionary:
        if dictionary[candidate] > c:
            return dictionary[candidate], candidate
        else:
            return c, previous
    else:
        return c, previous


def get_max(ngram_candidates):
    c1, maxc1 = 0, ""
    c2, maxc2 = 0, ""
    c3, maxc3 = 0, ""
    
    for candidate in ngram_candidates:
        if len(candidate.split()) == 3:
            c3, maxc3 = search_dict(candidate, maxc3, c3, gram3)
            
        if len(candidate.split()) == 2:
            c2, maxc2 = search_dict(candidate, maxc2, c2, gram2)
            
        if len(candidate.split()) == 1:
            c1, maxc1 = search_dict(candidate, maxc1, c1, gram1)
        
    if c3:
        return ngram_candidates[maxc3]
    else:
        if c2:
            return ngram_candidates[maxc2]
        else:
            return ngram_candidates[maxc1]


def lookup(sentence):
    sentence_list = sentence.lower().split()
    errores = detector(sentence_list)
    correction = []
    
    for x, (i, j) in enumerate(zip(sentence_list, errores)):
        
        if j:
            candidates = get_candidates(i)
            if candidates:
                grams = ngram(sentence_list, candidates, x)
                correct = get_max(grams)
                correction.append(correct)
            else:
               correction.append(i) 
        else:
            correction.append(i)
           
    return " ".join(correction)








