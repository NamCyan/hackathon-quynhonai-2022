from flask import Flask, request, jsonify
import config
import settings
from solver0 import ClassifyReviewSolver

review_solver = ClassifyReviewSolver(config)

app = Flask(__name__)

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    review_sentence = request.args.get('review_sentence')
    predict_results = review_solver.solve(review_sentence)

    output = {
        "review": review_sentence,
        "results": {}
      }
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = predict_results[count]

    return jsonify(output)

if __name__ == '__main__':
    app.run(host=settings.HOST, port=settings.PORT)