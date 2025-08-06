import pandas as pd
import os


corrige = "/home/gad/kurakura/french-slm/src/data/scholar-fr/pdf/corriges"
sujets = "/home/gad/kurakura/french-slm/src/data/scholar-fr/pdf/sujets"

corrige_names = os.listdir(corrige)
sujet_names = os.listdir(sujets)
sujet_names = set(sujet_names)
blacklist = set(["mathematiques","physique"])
pairs = []
for corrige in corrige_names: # find pairs
    if any(bad_word in corrige for bad_word in blacklist):
        continue

    sujet = corrige.replace("corrige", "sujet")
    if sujet in sujet_names:
        pairs.append([sujet, corrige])
    
    elif "-officiel" in corrige:
        sujet = corrige.replace("-officiel", "")
        if sujet in sujet_names:
            pairs.append([sujet, corrige])
    
    elif "-officiel" not in corrige:
        sujet = corrige.replace("corrige", "sujet-officiel")
        if sujet in sujet_names:
            pairs.append([sujet, corrige])
    
df = pd.DataFrame(pairs, columns=["sujet", "corrige"])
df.to_csv("/home/gad/kurakura/french-slm/src/data/scholar-fr/datas/bac/pdf_pairs.csv", index=False)
