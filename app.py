from flask import Flask, render_template, request, send_file
import os

from predict import extract_features
from heatmap import generate_heatmap
from clustering import generate_cluster_plot
from report import create_report

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def dashboard():

    feature_dim = None
    chart_values = None
    heatmap_path = None
    cluster_path = None
    img_path = None
    report_file = None
    score = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            # Extract features
            feature_dim, chart_values, features = extract_features(path)

            # Calculate abnormality score
            score = round(sum(features[:20]) / 20, 3)

            # Generate heatmap
            heatmap_path = generate_heatmap(path)

            # Generate clustering plot
            cluster_path = generate_cluster_plot(features)

            # Create PDF report
            report_file = create_report(path, feature_dim)

            img_path = path

    return render_template(
        "dashboard.html",
        feature_dim=feature_dim,
        chart_values=chart_values,
        img_path=img_path,
        heatmap_path=heatmap_path,
        cluster_path=cluster_path,
        report_file=report_file,
        score=score
    )


@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(REPORT_FOLDER, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)