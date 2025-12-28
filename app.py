from flask import Flask, request, render_template
import sys

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.transformer_predict_pipeline import TransformerPredictPipeline
from src.exception import CustomException

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def screen_resumes():
    results = None
    selected_model = "tfidf"

    if request.method == "POST":
        try:
            job_description = request.form.get("job_description")
            selected_model = request.form.get("model_type")

            if not job_description or len(job_description.strip()) < 20:
                raise ValueError("Job description is too short")

            if selected_model == "transformer":
                pipeline = TransformerPredictPipeline()
            else:
                pipeline = PredictPipeline()

            results = pipeline.predict(job_description, top_k=5)

        except Exception as e:
            raise CustomException(e, sys)

    return render_template(
        "index.html",
        results=results,
        selected_model=selected_model
    )


@app.route("/resume/<int:resume_id>")
def view_resume(resume_id):
    import pickle, os

    with open(os.path.join("artifacts", "resumes.pkl"), "rb") as f:
        resumes = pickle.load(f)

    return render_template(
        "resume.html",
        resume_id=resume_id,
        resume_text=resumes[resume_id]
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
