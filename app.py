import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from anthropic import Anthropic

st.set_page_config(page_title="Assistant IA", page_icon="⚙️", layout="centered")

load_dotenv()
client = Anthropic()


@st.cache_resource
def charger_modele():
    return SentenceTransformer("all-MiniLM-L6-v2")

modele = charger_modele()


def lire_pdf(fichier):
    lecteur = PdfReader(fichier)
    texte = ""
    for page in lecteur.pages:
        texte += page.extract_text()
    return texte


def decouper(texte, taille, chevauchement):
    morceaux = []
    for debut in range(0, len(texte), taille - chevauchement):
        morceaux.append(texte[debut:debut + taille])
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
        system="Tu es un assistant qui répond à partir des documents fournis. Si la réponse n'est pas dans les extraits, dis-le.",
        messages=[{"role": "user", "content": prompt}]
    )
    return reponse.content[0].text


# --- Barre latérale ---
with st.sidebar:
    st.header("Comment ça marche")
    st.markdown(
        "1. **Déposez** votre document PDF\n"
        "2. **Posez** votre question\n"
        "3. **Recevez** une réponse sourcée"
    )
    st.divider()
    st.caption("Les réponses s'appuient uniquement sur vos documents. Si l'information n'y figure pas, l'assistant le dit.")
    st.divider()
    st.markdown("**Damien** · Intégration IA pour PME")


# --- Page principale ---
st.title("Assistant IA")
st.markdown("##### Vos documents, vos réponses.")

fichier = st.file_uploader("Déposez votre document", type="pdf")

if fichier is None:
    st.caption("Commencez par déposer un document PDF : notice, procédure, règlement, contrat…")
else:
    with st.spinner("Lecture du document..."):
        texte = lire_pdf(fichier)
        morceaux = decouper(texte, 1000, 200)
    st.success(f"« {fichier.name} » chargé · {len(morceaux)} passages analysés")

    question = st.text_input("Votre question", placeholder="Ex. : quelle est la procédure en cas de panne ?")

    if st.button("Obtenir la réponse", type="primary"):
        if question == "":
            st.warning("Écrivez d'abord une question.")
        else:
            with st.spinner("Recherche dans le document..."):
                extraits = trouver_morceaux(question, morceaux, 3)
                reponse = demander_claude(question, extraits)

            st.subheader("Réponse")
            with st.container(border=True):
                st.markdown(reponse)

            st.subheader("Sources")
            for numero, extrait in enumerate(extraits, start=1):
                with st.expander(f"Extrait {numero}"):
                    st.text(extrait)
                    