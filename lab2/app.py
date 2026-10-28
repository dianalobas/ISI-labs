import pandas as pd
from flask import Flask, jsonify, request
from recommender import recommend

app = Flask(__name__)

@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "podaj parametr ?title=..."}), 400
    n = request.args.get("top_n", type=int) if "top_n" in request.args else 5
    if n is None or not 1 <= n <= 10:
        return jsonify({"error": "top_n musi byc z zakresu 1-10"}), 400

    target_genre = request.args.get("genre") or request.args.get("genres")

    recs = recommend(title, top_n=n)
    if recs is None:
        return jsonify({"error": "nie znam takiego filmu"}), 404

    if target_genre:
        target_genre_clean = target_genre.strip().lower()

        df = pd.DataFrame(recs)

        genre_column = "genres" if "genres" in df.columns else "genre"

        if genre_column in df.columns:
            df_filtered = df[
                df[genre_column].str.contains(target_genre_clean, case=False, na=False)
            ]

            recs = df_filtered.to_dict(orient="records")
        else:
            recs = []
    return jsonify({"title": title, "count": len(recs), "recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)