# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:01:54 2022

@author: Irving
"""

import os
import pandas as pd
import numpy as np
import pickle

from spell_cat import lookup

path = os.getcwd()

with open(os.path.join(path, "Comentarios para ML.xlsx"), "rb") as g:
    data = pd.read_excel(g)
    
with open(os.path.join(path, "linearsvc.pickle"), "rb") as g:
    svc = pickle.load(g)
       

use = data[["ID de respuesta", "Input"]]
use["Input"] = use["Input"].str.replace(r"[^\w\s]", "")
use["Input"] = use["Input"].replace("", np.nan)

df = use.dropna()
df["fixed"] = df["Input"].apply(lookup)

df["pred"] = svc.predict(df["fixed"])

pred = pd.merge(data, df[["ID de respuesta", "pred"]],
                how= "left", on= "ID de respuesta")

with open(os.path.join(path, "Palancas ML.xlsx"), "wb") as f:
    pred[["ID de respuesta", "Input", "pred"]].to_excel(f, "predict", index= False)






    