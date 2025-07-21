import os
import fitz  # PyMuPDF
import json
import google.generativeai as genai

# Configuration de l’API Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Lecture du texte PDF
def extract_text(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

# Appel Gemini (simple wrapper)
def call_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERREUR: {e}"

# Étape 1 : Détection automatique des questions (sans regex)
def extract_questions_with_gemini(subject_text):
    prompt = f"""
You are given the full text of a French high school exam.

Task:
- Extract all actual **student questions** that require an answer.
- Do not include titles, sections, instructions, or metadata.
- Questions can be numbered in many ways (1., 1), 1a, Q1, etc.) — detect all valid forms.
- Return a **JSON list of strings**: one entry per question.
- Do not rephrase. Copy questions exactly as they appear.

Exam text:
{subject_text}
"""
    result = call_gemini(prompt)
    try:
        return json.loads(result)
    except Exception as e:
        print("Erreur JSON dans l'extraction des questions:", e)
        return []

# Étape 2 : Filtrer les questions sans image ou schéma
def filter_questions(questions):
    joined = "\n".join(questions)
    prompt = f"""
You are given a list of French high school exam questions. 
Return **only** the ones that can be answered **without any diagram, image, drawing, or graph**. 
List them exactly as they appear. Do not explain or rephrase.

List:
{joined}
"""
    result = call_gemini(prompt)
    return [q.strip() for q in result.split("\n") if q.strip()]

# Étape 3 : Ajouter le contexte d’énoncé à chaque question
def add_context_to_questions(subject_text, questions):
    prompt = f"""
The following is the full text of a French high school exam:
----
{subject_text}
----

For each question below, extract its relevant context or introductory description from the subject (such as problem description or setup).
Do not summarize or rewrite. 
Return a JSON list of dictionaries with keys: "question" and "context".

Questions:
{json.dumps(questions, ensure_ascii=False)}
"""
    result = call_gemini(prompt)
    try:
        return json.loads(result)
    except:
        print("Erreur JSON dans l'étape 2.")
        return []

# Étape 4 : Extraire la réponse exacte du corrigé
def extract_answers_with_context(contexted_questions, correction_text):
    prompt = f"""
You are extracting exact answers (in French) from the correction of a French high school exam.

Rules:
- Use only what is explicitly written in the correction text below.
- Return the answer **exactly** as it appears, in French.
- Use LaTeX formatting for equations.
- If an image or diagram is needed, return exactly: "NON TRAITÉ - nécessite un schéma ou une image."

Correction:
{correction_text}

Questions with context:
{json.dumps(contexted_questions, ensure_ascii=False)}
"""
    result = call_gemini(prompt)
    try:
        return json.loads(result)
    except:
        print("Erreur JSON dans l'étape 3.")
        return []

# Étape 5 : Ajouter les réponses précédentes nécessaires
def enrich_context_with_previous(answers_with_context):
    prompt = f"""
Some questions below may require previous answers to make sense (multi-step problems).

For each question, if a previous answer is needed to understand or solve it, add it to the context.

Return a JSON list of updated items with keys: "question", "context", "reponse"

Entries:
{json.dumps(answers_with_context, ensure_ascii=False)}
"""
    result = call_gemini(prompt)
    try:
        return json.loads(result)
    except:
        print("Erreur JSON dans l'étape 4.")
        return []

# Pipeline principal
def full_pipeline(subject_pdf, correction_pdf):
    print("Extraction du texte...")
    subject_text = extract_text(subject_pdf)
    correction_text = extract_text(correction_pdf)

    print("Étape 1 : Détection des questions...")
    raw_questions = extract_questions_with_gemini(subject_text)

    print("Étape 2 : Filtrage des questions sans schéma...")
    filtered_questions = filter_questions(raw_questions)

    print("Étape 3 : Ajout du contexte d’énoncé...")
    contexted_questions = add_context_to_questions(subject_text, filtered_questions)

    print("Étape 4 : Extraction des réponses depuis le corrigé...")
    answered = extract_answers_with_context(contexted_questions, correction_text)

    print("Étape 5 : Ajout du contexte des réponses précédentes...")
    final_result = enrich_context_with_previous(answered)

    return final_result

# Sauvegarde JSON
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Script exécutable
if __name__ == "__main__":
    sujet_pdf = "sujet.pdf"         
    correction_pdf = "corrige.pdf"  
    output_path = "resultats_final.json"

    result = full_pipeline(sujet_pdf, correction_pdf)
    save_json(result, output_path)
    print(f"\n✅ Terminé ! Résultats sauvegardés dans : {output_path}")
