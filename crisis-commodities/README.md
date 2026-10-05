# Crisis and Commodities — Adam Azouz

Code extrait de l'annexe D du mémoire, pages 81 à 92.

## Fichiers

- `thesis_outputs.py` : tableaux statistiques et graphiques de corrélations glissantes ; télécharge les données depuis Yahoo Finance.
- `dcc_analysis.py` : corrélation dynamique EWMA sur résidus GARCH ; demande des données Excel locales.
- `requirements.txt` : bibliothèques Python à installer.

## Exécuter dans GitHub Codespaces

Python 3.10 ou plus récent est nécessaire.

Dans le terminal, exécuter ces commandes une par une :

```bash
python -m pip install -r requirements.txt
python thesis_outputs.py
```

Les résultats sont enregistrés dans `figures/` et `tables/`.

## Analyse DCC (facultative)

Ajouter dans un dossier `raw_data/` les fichiers suivants :

- `gold.xlsx`
- `silver.xlsx`
- `crude_oil.xlsx`
- `copper.xlsx`
- `s&p_500.xlsx`

Chaque fichier doit avoir les dates dans la première colonne et une colonne de prix `Adj Close` ou `Close`. Les fichiers de données ne sont pas inclus. Le premier script du PDF ne crée pas ces fichiers Excel, malgré le message du second script.

Ensuite :

```bash
python dcc_analysis.py
```

## État de cette extraction

Les en-têtes, pieds de page, indentations et retours de ligne du PDF ont été corrigés. La syntaxe des deux scripts a été vérifiée. Ils n'ont pas été exécutés sur les données réelles ; la disponibilité des données Yahoo Finance et la compatibilité des dépendances restent à vérifier.

La logique de calcul du mémoire est conservée. Le tableau A.1 contient des valeurs d'exemple codées dans le script, pas une vérification Bloomberg réelle. L'analyse nommée DCC utilise une mise à jour EWMA ; ce n'est pas une estimation complète d'un modèle DCC-GARCH.
