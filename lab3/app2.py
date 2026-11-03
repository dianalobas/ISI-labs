from flask import Flask, jsonify, request, send_from_directory
import os

# Importujemy z pliku recommender.py w tym samym folderze
from recommender import matrix, recommend, similar_users

app = Flask(__name__)


@app.get("/")
def home():
    # Jeśli masz plik index.html w folderze, zwróci go, w przeciwnym razie zwróci komunikat
    if os.path.exists("index.html"):
        return send_from_directory(".", "index.html")
    return "Lab 03 - Systemy Rekomendacji działa! Użyj /api/users lub /api/recommend?user=..."


@app.route("/api/recommend")
def api_recommend():
    user = request.args.get("user", "").strip()
    if not user:
        return jsonify({"error": "podaj parametr ?user=..."}), 400

    count = request.args.get("top_n", type=int) if "top_n" in request.args else 5
    if "top_n" in request.args and (count is None or not 1 <= count <= 10):
        return jsonify({"error": "top_n musi być od 1 do 10"}), 400

    recs = recommend(user, top_n=count)
    if recs is None:
        return jsonify({
            "error": "nie znam takiego użytkownika",
            "known_users": sorted(matrix.index.tolist())
        }), 404

    return jsonify({"user": user, "count": len(recs), "recommendations": recs})


@app.route("/api/users")
def api_users():
    return jsonify({"count": len(matrix.index), "users": sorted(matrix.index.tolist())})


@app.route("/api/similar")
def api_similar():
    user = request.args.get("user", "").strip()
    if not user:
        return jsonify({"error": "podaj parametr ?user=..."}), 400
    s = similar_users(user)
    if s is None:
        return jsonify({"error": "nie znam takiego użytkownika"}), 404
    return jsonify({"user": user, "similar": s})


if __name__ == "__main__":
    app.run(debug=True, port=5000)