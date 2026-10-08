# Assistant IA — Vos documents, vos réponses

Assistant qui répond aux questions à partir de vos propres documents PDF (notices, procédures, règlements, contrats), avec les sources affichées et sans invention.

## Le problème
Dans beaucoup d'entreprises, l'information existe mais personne ne la retrouve : documentation éparpillée, notices de centaines de pages, savoir détenu par quelques anciens.

## Ce que fait l'outil
1. **Lecture** du document PDF déposé
2. **Découpage** en passages qui se chevauchent, pour ne rien couper en deux
3. **Recherche sémantique** des 3 passages les plus pertinents (par le sens, pas par mots-clés)
4. **Réponse** rédigée par Claude (Anthropic) à partir de ces seuls passages, avec les extraits sources affichés

Si l'information n'est pas dans le document, l'assistant le dit au lieu d'inventer.

## Technologies
Python · Streamlit · pypdf · sentence-transformers (embeddings) · API Claude (Anthropic) · architecture RAG (Retrieval-Augmented Generation)

## Lancer l'application
1. Installer les dépendances : `pip install -r requirements.txt`
2. Créer un fichier `.env` contenant votre clé : `ANTHROPIC_API_KEY=...`
3. Lancer : `streamlit run app.py`

## Auteur
Damien — 17 ans de maintenance technique sur le terrain, aujourd'hui spécialisé dans l'intégration de l'IA pour les PME : diagnostic d'abord, outil concret ensuite.