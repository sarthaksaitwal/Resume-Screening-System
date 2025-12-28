from flask import Flask, request, render_template
import sys

from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException

application = Flask(__name__)
app = application


@app.route("/", methods=["GET", "POST"])
def screen_resumes():
    results = None

    if request.method == "POST":
        try:
            job_description = request.form.get("job_description")

            if not job_description or len(job_description.strip()) < 20:
                raise ValueError("Job description is too short")

            pipeline = PredictPipeline()
            results = pipeline.predict(job_description, top_k=5)

        except Exception as e:
            raise CustomException(e, sys)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
