import os
import json
import pandas as pd
import re
# script to gather all questions and answers from JSON files in one file and clean if badly formatted or escaped

dossier_json = r"..." #folder with all the json files, each containing a question and its answer
reponses = os.listdir(dossier_json)

questions_combinees = {
    'questions': [],
    'context': [],
    'responses': [],
    'subject': []
}

nb_questions = 0

def safe_serialize(obj):
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)
    return str(obj) if obj is not None else ""

for fichier in reponses:
    chemin = os.path.join(dossier_json, fichier)
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)

   
        if isinstance(data, dict) and "result" in data:
            subject = data.get("subject", "")
            result_data = data["result"]
            

            if isinstance(result_data, str):
                try:
                    data_list = json.loads(result_data)
                    if not isinstance(data_list, list):
                        print(f"⚠️ 'result' parsé n'est pas une liste dans : {fichier}")
                        continue
                except json.JSONDecodeError as e:
                    print(f"⚠️ Tentative de correction des échappements dans {fichier}")
                    try:

                        corrected_data = result_data.replace('\\\\', '\\').replace('\\"', '"')
 
                        corrected_data = corrected_data.replace('\\n', '\n').replace('\\t', '\t')
                        corrected_data = corrected_data.replace('\\r', '\r').replace('\\f', '\f')
                        corrected_data = corrected_data.replace('\\b', '\b')
   
                        corrected_data = corrected_data.replace('\\cdot', '\\\\cdot')
                        corrected_data = corrected_data.replace('\\times', '\\\\times')
                        corrected_data = corrected_data.replace('\\theta', '\\\\theta')
                        corrected_data = corrected_data.replace('\\phi', '\\\\phi')
                        corrected_data = corrected_data.replace('\\alpha', '\\\\alpha')
                        corrected_data = corrected_data.replace('\\omega', '\\\\omega')
                        corrected_data = corrected_data.replace('\\Delta', '\\\\Delta')
                        corrected_data = corrected_data.replace('\\pi', '\\\\pi')
                        corrected_data = corrected_data.replace('\\vec', '\\\\vec')
                        corrected_data = corrected_data.replace('\\text', '\\\\text')
                        corrected_data = corrected_data.replace('\\Rightarrow', '\\\\Rightarrow')
                        
                        data_list = json.loads(corrected_data)
                        if not isinstance(data_list, list):
                            print(f"⚠️ 'result' parsé n'est pas une liste dans : {fichier}")
                            continue
                        print(f"✅ Correction réussie pour {fichier}")
                    except json.JSONDecodeError as e2:
                        print(f"⚠️ Impossible de parser même après correction dans {fichier} : {e2}")
        
                        try:
                            import ast
                            data_list = ast.literal_eval(result_data)
                            if not isinstance(data_list, list):
                                print(f"⚠️ 'result' parsé avec ast n'est pas une liste dans : {fichier}")
                                continue
                            print(f"✅ Parsing avec ast réussi pour {fichier}")
                        except Exception as e3:
                            print(f"⚠️ Tentative d'extraction manuelle pour {fichier}")
                            try:

                                patterns = [
                                    r'"question"\s*:\s*"([^"]*(?:\\.[^"]*)*)".*?"reponse"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                    r'"question"\s*:\s*"([^"]*(?:\\.[^"]*)*)".*?"fr_answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                    r'"question"\s*:\s*"([^"]*(?:\\.[^"]*)*)".*?"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                                    r'"question"\s*:\s*"([^"]*(?:\\.[^"]*)*)".*?"response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                                ]
                                
                                matches = []
                                for pattern in patterns:
                                    pattern_matches = re.findall(pattern, result_data, re.DOTALL)
                                    if pattern_matches:
                                        matches.extend(pattern_matches)
                                        break  
                                
                                if matches:
                                    data_list = []
                                    for question, reponse in matches:
    
                                        question_clean = question.replace('\\"', '"').replace('\\\\', '\\')
                                        reponse_clean = reponse.replace('\\"', '"').replace('\\\\', '\\')
                                        
                                        data_list.append({
                                            "question": question_clean,
                                            "reponse": reponse_clean,
                                            "context": None
                                        })
                                    
                                    if data_list:
                                        print(f"✅ Extraction manuelle réussie pour {fichier} ({len(data_list)} entrées)")
                                    else:
                                        print(f"⚠️ Aucune donnée extraite pour {fichier}")
                                        continue
                                else:
                                    print(f"⚠️ Aucun pattern trouvé pour {fichier}")
                     
                                    simple_pattern = r'\{[^}]*"question"[^}]*\}'
                                    simple_matches = re.findall(simple_pattern, result_data)
                                    if simple_matches:
                                        print(f"⚠️ Trouvé {len(simple_matches)} objets possibles, mais structure complexe")
                                        print(f"⚠️ Contenu des premiers caractères: {result_data[:200]}...")
                                    continue
                                    
                            except Exception as e4:
                                print(f"⚠️ Échec complet du parsing pour {fichier} : {e4}")
                                print(f"⚠️ Contenu des premiers caractères: {result_data[:200]}...")
                                continue

            elif isinstance(result_data, list):
                data_list = result_data
            else:
                print(f"⚠️ 'result' n'est ni une string ni une liste dans : {fichier}")
                continue


        elif isinstance(data, list):
            subject = ""
            data_list = data

        else:
            print(f"⚠️ Format inattendu dans : {fichier}")
            continue


        def get_response_field(sample):
            """Récupère la réponse en essayant différents noms de champs"""
            for field_name in ['reponse', 'fr_answer', 'answer', 'response']:
                if sample.get(field_name):
                    return sample.get(field_name)
            return None

        filtered = [sample for sample in data_list if sample.get("question") and get_response_field(sample)]

        for sample in filtered:
            nb_questions += 1
            questions_combinees['questions'].append(sample.get('question'))
            questions_combinees['context'].append(safe_serialize(sample.get('context')))
            questions_combinees['responses'].append(safe_serialize(get_response_field(sample)))
            questions_combinees['subject'].append(subject)

    except Exception as e:
        print(f"Error with file {fichier} : {e}")

print(f"Total number of valid questions : {nb_questions}")

df = pd.DataFrame(questions_combinees)
df.to_csv("questions_combinees.csv", index=False)