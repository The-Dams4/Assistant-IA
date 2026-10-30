import streamlit as st
from pypdf import PdfReader
from docx import Document
from docx.table import Table
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from anthropic import Anthropic
 
st.set_page_config(page_title="Assistant IA", page_icon="⚙️", layout="centered")
 
load_dotenv()
client = Anthropic()
 
 
@st.cache_resource
def charger_modele():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
 
modele = charger_modele()
 
 
def lire_pdf(fichier):
    lecteur = PdfReader(fichier)
    texte = ""
    for page in lecteur.pages:
        texte += page.extract_text() + "\n"
    texte = texte.replace("\t", " ")
    return texte
 
 
def decouper(texte, taille, chevauchement):
    morceaux = []
    for debut in range(0, len(texte), taille - chevauchement):
        morceaux.append(texte[debut:debut + taille])
    return morceaux
 
 
def lire_word(fichier):
    doc = Document(fichier)
    texte = ""
    for element in doc.iter_inner_content():
        if isinstance(element, Table):
            for ligne in element.rows:
                texte_ligne = ""
                for cellule in ligne.cells:
                    texte_ligne += cellule.text + " | "
                texte += texte_ligne + "\n"
        else:
            texte += element.text + "\n"
    return texte
 
def est_un_titre(ligne):
    return ligne[0:1].isdigit() and ligne[1:3] == ". " and len(ligne.strip()) > 3
 
 
def decouper_par_sections(texte):
    lignes = texte.split("\n")
    sections = []
    section_en_cours = ""
    for ligne in lignes:
        if est_un_titre(ligne) and section_en_cours != "":
            sections.append(section_en_cours)
            section_en_cours = ""
        section_en_cours += ligne + "\n"
    sections.append(section_en_cours)
    return sections
 
 
def redecouper_section(section, taille_max):
    lignes = section.split("\n")
    titre = lignes[0]
    morceaux = []
    morceau_en_cours = ""
    for ligne in lignes:
        if len(morceau_en_cours) + len(ligne) > taille_max and morceau_en_cours != "":
            morceaux.append(morceau_en_cours)
            morceau_en_cours = titre + " (suite)\n"
        morceau_en_cours += ligne + "\n"
    morceaux.append(morceau_en_cours)
    return morceaux
 
 
def decouper_document(texte):
    sections = decouper_par_sections(texte)
    if len(sections) < 3:
        return decouper(texte, 1000, 200)
    morceaux = []
    for section in sections:
        if len(section) <= 1200:
            morceaux.append(section)
        else:
            morceaux += redecouper_section(section, 800)
    return morceaux
 
 
def trouver_morceaux(question, morceaux, nombre):
    vecteurs_morceaux = modele.encode(morceaux)
    vecteur_question = modele.encode(question)
    similarites = util.cos_sim(vecteur_question, vecteurs_morceaux)[0]
    meilleurs_index = similarites.argsort(descending=True)[:nombre]
    resultat = []
    for index in meilleurs_index:
        resultat.append(morceaux[int(index)])
    return resultat
 
 
def demander_claude(question, extraits):
    contexte = "\n---\n".join(extraits)
    prompt = f"Voici des extraits de document :\n{contexte}\nQuestion : {question}\nRéponds en t'appuyant sur ces extraits."
    reponse = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system="Tu es un assistant qui répond à partir des documents fournis. Chaque extrait commence par le nom de son document entre crochets : indique de quel document vient ta réponse. Si la réponse n'est pas dans les extraits, dis-le. N'utilise pas de titres Markdown (#) : mets seulement les mots importants en gras.",
        messages=[{"role": "user", "content": prompt}]
    )
    return reponse.content[0].text
 
 
# --- Barre latérale ---
with st.sidebar:
    st.header("Comment ça marche")
    st.markdown(
        "1. **Déposez** vos documents PDF ou Word\n"
        "2. **Posez** votre question\n"
        "3. **Recevez** une réponse sourcée"
    )
    st.divider()
    st.caption("Les réponses s'appuient uniquement sur vos documents. Si l'information n'y figure pas, l'assistant le dit.")
    st.divider()
    st.markdown("**DY Solutions** · IA et Automatisation pour artisans et PME")
 
 
# --- Page principale ---
st.title("Assistant IA")
st.markdown("##### Vos documents, vos réponses.")
 
mot_de_passe = st.text_input("Mot de passe de démonstration", type="password")
if mot_de_passe != st.secrets["APP_PASSWORD"]:
    st.caption("Accès réservé : saisissez le mot de passe qui vous a été communiqué.")
    st.stop()
 
fichiers = st.file_uploader("Déposez vos documents", type=["pdf", "docx"], accept_multiple_files=True)
 
 
if len(fichiers) == 0:
    st.caption("Commencez par déposer un ou plusieurs documents PDF ou Word : notices, procédures, règlements, contrats…")
else:
    morceaux = []
    with st.spinner("Lecture des documents..."):
        for fichier in fichiers:
            if fichier.name.endswith(".docx"):
                texte = lire_word(fichier)
            else:
                texte = lire_pdf(fichier)
            for morceau in decouper_document(texte):
                morceaux.append(f"[{fichier.name}]\n" + morceau)
    st.success(f"{len(fichiers)} document(s) chargé(s) · {len(morceaux)} passages analysés")
 
    question = st.text_input("Votre question", placeholder="Ex. : quelle est la procédure en cas de panne ?")
 
    if st.button("Obtenir la réponse", type="primary"):
        if question == "":
            st.warning("Écrivez d'abord une question.")
        else:
            with st.spinner("Recherche dans les documents..."):
                extraits = trouver_morceaux(question, morceaux, 4)
                reponse = demander_claude(question, extraits)
 
            st.subheader("Réponse")
            with st.container(border=True):
                st.markdown(reponse)
 
            st.subheader("Sources")
            for numero, extrait in enumerate(extraits, start=1):
                with st.expander(f"Extrait {numero}"):
                    st.write(" ".join(extrait.split()))
