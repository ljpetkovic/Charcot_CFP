import pandas as pd
from pathlib import Path


# =========================
# Fichiers d'entrée
# =========================

CHARCOT_CSV = Path("frequences_detaillees_charcot.csv")
AUTRES_CSV = Path("frequences_detaillees_autres.csv")


# =========================
# Fichiers de sortie
# =========================

OUTPUT_CSV = Path("resultats_ppm_globaux_tous_termes.csv")
OUTPUT_CHARCOT_TXT = Path("resultats_charcot_ppm_globaux.txt")
OUTPUT_AUTRES_TXT = Path("resultats_autres_ppm_globaux.txt")


def calculer_ppm_globaux(path, nom_corpus):
    """
    Calcule les fréquences normalisées globales en ppm
    pour tous les termes d'un corpus.

    Le fichier d'entrée contient des fréquences par année :
    - date
    - total_tokens
    - expression
    - occurrences
    - freq_ppm

    Attention :
    on ne somme pas les freq_ppm annuels.
    On somme d'abord les occurrences de chaque terme sur toutes les années,
    puis on divise par le nombre total de tokens du corpus.
    """

    df = pd.read_csv(path)

    # Vérification des colonnes nécessaires.
    colonnes_attendues = {"date", "total_tokens", "expression", "occurrences"}
    colonnes_manquantes = colonnes_attendues - set(df.columns)

    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans {path.name} : {colonnes_manquantes}"
        )

    # total_tokens est répété pour chaque terme d'une même année.
    # Il faut donc ne compter chaque année qu'une seule fois.
    total_tokens_corpus = (
        df[["date", "total_tokens"]]
        .drop_duplicates()
        ["total_tokens"]
        .sum()
    )

    # Somme des occurrences de chaque terme sur toutes les années.
    resultats = (
        df.groupby("expression", as_index=False)
        .agg(occurrences_globales=("occurrences", "sum"))
    )

    # Calcul du ppm global.
    resultats[f"ppm_{nom_corpus}"] = (
        resultats["occurrences_globales"] / total_tokens_corpus * 1_000_000
    )

    # Renommage pour garder la trace du corpus.
    resultats = resultats.rename(
        columns={
            "occurrences_globales": f"occurrences_{nom_corpus}"
        }
    )

    # Ajout du total de tokens utilisé pour le calcul.
    resultats[f"tokens_{nom_corpus}"] = total_tokens_corpus

    # Tri par fréquence décroissante.
    resultats = resultats.sort_values(
        by=f"ppm_{nom_corpus}",
        ascending=False
    )

    return resultats, total_tokens_corpus


# =========================
# Calcul pour chaque corpus
# =========================

charcot, tokens_charcot = calculer_ppm_globaux(
    CHARCOT_CSV,
    "charcot"
)

autres, tokens_autres = calculer_ppm_globaux(
    AUTRES_CSV,
    "autres"
)


# =========================
# Fusion des deux corpus
# =========================

comparaison = pd.merge(
    charcot,
    autres,
    on="expression",
    how="outer"
)

# Les valeurs manquantes correspondent aux termes absents d'un corpus.
comparaison["occurrences_charcot"] = comparaison["occurrences_charcot"].fillna(0).astype(int)
comparaison["occurrences_autres"] = comparaison["occurrences_autres"].fillna(0).astype(int)

comparaison["ppm_charcot"] = comparaison["ppm_charcot"].fillna(0)
comparaison["ppm_autres"] = comparaison["ppm_autres"].fillna(0)

comparaison["tokens_charcot"] = comparaison["tokens_charcot"].fillna(tokens_charcot).astype(int)
comparaison["tokens_autres"] = comparaison["tokens_autres"].fillna(tokens_autres).astype(int)


# =========================
# Colonnes supplémentaires utiles
# =========================

# Différence de fréquence normalisée.
# Valeur positive : terme plus fréquent dans Charcot.
# Valeur négative : terme plus fréquent dans Autres.
comparaison["diff_ppm_charcot_moins_autres"] = (
    comparaison["ppm_charcot"] - comparaison["ppm_autres"]
)

# Ratio Charcot / Autres avec une petite correction
# pour éviter les divisions par zéro.
epsilon = 0.5

comparaison["ratio_charcot_autres"] = (
    (comparaison["occurrences_charcot"] + epsilon) / tokens_charcot
) / (
    (comparaison["occurrences_autres"] + epsilon) / tokens_autres
)

# Tri final par fréquence décroissante dans Charcot.
comparaison = comparaison.sort_values(
    by=["ppm_charcot", "expression"],
    ascending=[False, True]
)


# =========================
# Sauvegarde du tableau complet
# =========================

comparaison.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)


# =========================
# Sauvegarde de deux listes simples
# =========================

# Format :
# expression;ppm

charcot_liste = comparaison[comparaison["ppm_charcot"] > 0].sort_values(
    by=["ppm_charcot", "expression"],
    ascending=[False, True]
)

autres_liste = comparaison[comparaison["ppm_autres"] > 0].sort_values(
    by=["ppm_autres", "expression"],
    ascending=[False, True]
)

with open(OUTPUT_CHARCOT_TXT, "w", encoding="utf-8") as f:
    for _, row in charcot_liste.iterrows():
        f.write(f"{row['expression']};{row['ppm_charcot']:.6f}\n")

with open(OUTPUT_AUTRES_TXT, "w", encoding="utf-8") as f:
    for _, row in autres_liste.iterrows():
        f.write(f"{row['expression']};{row['ppm_autres']:.6f}\n")


# =========================
# Affichage de contrôle
# =========================

print("Fichiers générés :")
print(OUTPUT_CSV)
print(OUTPUT_CHARCOT_TXT)
print(OUTPUT_AUTRES_TXT)

print()
print("Total tokens Charcot :", tokens_charcot)
print("Total tokens Autres  :", tokens_autres)

print()
print("Aperçu des résultats :")
print(
    comparaison[
        [
            "expression",
            "occurrences_charcot",
            "tokens_charcot",
            "ppm_charcot",
            "occurrences_autres",
            "tokens_autres",
            "ppm_autres",
            "diff_ppm_charcot_moins_autres",
            "ratio_charcot_autres",
        ]
    ].head(10)
)