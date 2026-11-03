import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Wczytanie danych i zbudowanie macierzy (12 osób x 24 filmy)
# Jeśli masz Excela, zamień na: pd.read_excel("ratings.xlsx")
df = pd.read_csv("ratings.csv")
matrix = df.pivot_table(index="user", columns="title", values="rating")

# 2. Centrowanie ocen (odejmujemy średnią każdej osoby, braki NaN zamieniamy na 0)
means = matrix.mean(axis=1)
centered = matrix.sub(means, axis=0).fillna(0)

# Liczymy podobieństwo kosinusowe między osobami
similarity_matrix = pd.DataFrame(
    cosine_similarity(centered),
    index=matrix.index,
    columns=matrix.index
)


# 3. Funkcja recommend(user, top_n=5, k=3)
def recommend(user, top_n=5, k=3):
    if user not in matrix.index:
        return None

    # Pobieramy k najbliższych sąsiadów (odrzucając samego siebie)
    neighbors = similarity_matrix[user].drop(user).sort_values(ascending=False).head(k)

    scores = {}
    for movie in matrix.columns:
        # Pominąć filmy, które użytkownik już ocenił
        if not pd.isna(matrix.loc[user, movie]):
            continue

        num, den = 0.0, 0.0
        for other, w in neighbors.items():
            r = matrix.loc[other, movie]
            if not pd.isna(r):
                # Wzór: waga * odchylenie sąsiada od jego średniej
                num += w * (r - means[other])
                den += abs(w)

        if den > 0:
            # Przewidywana ocena = średnia użytkownika + skorygowana suma odchyleń
            predicted_score = means[user] + (num / den)
            # Ograniczamy do skali 1.0 - 5.0
            scores[movie] = max(1.0, min(5.0, predicted_score))

    # Wybieramy top_n filmów z najwyższą przewidywaną oceną
    best_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for title, score in best_movies:
        # Sprawdzamy, którzy sąsiedzi faktycznie ocenili ten film (pole 'because')
        because_list = [
            other for other in neighbors.index
            if not pd.isna(matrix.loc[other, title])
        ]
        results.append({
            "title": title,
            "predicted": round(float(score), 2),
            "because": because_list
        })

    return results