import pandas as pd
import tiktoken  


df = pd.read_csv(r"/home/gad/kurakura/french-slm/src/data/bac-fr/datas/questions_combinees.csv")

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Choisir les colonnes à analyser
text_columns = ['question', 'context', 'responses']  
# Fonction pour compter les tokens dans une ligne
def count_tokens(row):
    total = 0
    for col in text_columns:
        if pd.notna(row.get(col)):
            total += len(encoding.encode(str(row[col])))
    return total

# Appliquer à chaque ligne
df['token_count'] = df.apply(count_tokens, axis=1)

# Compter le total
total_tokens = df['token_count'].sum()

print(f"Nombre total de tokens dans le dataset : {total_tokens}")
print(f"Moyenne de tokens par entrée : {df['token_count'].mean():.2f}")