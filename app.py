"""
Auto Data Analyst Agent - Flask Web Application
Run: python app.py
"""
import os
import json
import io
import base64
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

from agent.loader import load_data, DataLoadError
from agent.command_parser import CommandParser
from agent.insights import InsightsEngine
from agent.report_generator import ReportGenerator
from agent.planner import PlannerAgent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {
    "csv", "xlsx", "xls", "json", "txt", "pdf", "db", "sqlite", "sqlite3",
    "mp4", "avi", "mov", "mkv",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Upload a file and run a command on it."""
    file = request.files.get("file")
    command = request.form.get("command", "").strip().lower()

    if not file or file.filename == "":
        return jsonify({"type": "error", "message": "Please upload a data file."})
    if not allowed_file(file.filename):
        return jsonify({"type": "error", "message": "Unsupported file type."})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        data = load_data(filepath)
    except DataLoadError as e:
        return jsonify({"type": "error", "message": str(e)})
    except Exception as e:
        return jsonify({"type": "error", "message": f"Failed to load file: {str(e)}"})

    parser = CommandParser(data)
    result = parser.parse(command)
    result["filename"] = filename
    result["file_type"] = os.path.splitext(filename)[1].lstrip(".")
    return jsonify(result)


@app.route("/api/report", methods=["POST"])
def generate_report_pdf():
    """Generate and download an executive PDF report for the uploaded data."""
    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"type": "error", "message": "Please upload a data file."})
    if not allowed_file(file.filename):
        return jsonify({"type": "error", "message": "Unsupported file type."})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        data = load_data(filepath)
    except DataLoadError as e:
        return jsonify({"type": "error", "message": str(e)})
    except Exception as e:
        return jsonify({"type": "error", "message": f"Failed to load file: {str(e)}"})

    try:
        engine = InsightsEngine(data)
        report_data = engine.generate_report()
        if "error" in report_data:
            return jsonify({"type": "error", "message": report_data["error"]})
        gen = ReportGenerator()
        pdf_bytes = gen.generate_pdf(report_data)
    except Exception as e:
        return jsonify({"type": "error", "message": f"Report generation failed: {str(e)}"})

    report_filename = os.path.splitext(filename)[0] + "_report.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=report_filename,
    )


@app.route("/api/command", methods=["POST"])
def run_command():
    """Run a command on previously uploaded data (data passed in request)."""
    payload = request.get_json(silent=True) or {}
    command = payload.get("command", "").strip().lower()
    data = payload.get("data")

    if data is None:
        return jsonify({"type": "error", "message": "No data provided. Upload a file first."})

    parser = CommandParser(data)
    result = parser.parse(command)
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat_with_data():
    """Chat with the data using natural language.

    Uses the Planner Agent to orchestrate the appropriate specialized agent(s)
    based on the user's question. Returns a human-readable answer plus any
    structured result (charts, insights, tables, etc.).
    """
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()
    data = payload.get("data")

    if not message:
        return jsonify({"type": "error", "message": "Please enter a question."})
    if data is None:
        return jsonify({"type": "error", "message": "No data provided. Upload a file first."})

    # Use the command parser to interpret the natural-language message
    parser = CommandParser(data)
    result = parser.parse(message.lower())

    # Build a chat-friendly response
    response = {
        "type": result.get("type", "error"),
        "message": message,
        "answer": _build_chat_answer(result),
        "result": result,
    }
    return jsonify(response)


def _build_chat_answer(result):
    """Build a concise natural-language answer from a command result."""
    rtype = result.get("type")

    if rtype == "error":
        return result.get("message", "I couldn't analyze that.")

    if rtype == "summary":
        reps = result.get("reports", [])
        if reps:
            rep = reps[0]
            return (
                f"Your dataset has {rep['shape']['rows']} rows and "
                f"{rep['shape']['columns']} columns across "
                f"{len(rep['columns'])} columns: {', '.join(rep['columns'][:5])}."
            )
        return "Here is the summary of your data."

    if rtype == "describe":
        return "Here are the statistical summaries for your numeric columns."

    if rtype == "nulls":
        reps = result.get("reports", [])
        if reps:
            rep = reps[0]
            return (
                f"Found {rep['total_null_cells']} missing cells "
                f"({rep['null_percentage']}% of the data) across your dataset."
            )
        return "No missing value analysis available."

    if rtype == "correlation":
        return "Here is the correlation matrix showing relationships between numeric columns."

    if rtype == "head":
        return "Here are the first 10 rows of your data."

    if rtype == "unique":
        return "Here are the unique value counts for each column."

    if rtype == "chart":
        return "Here's a visualization of your data."

    if rtype == "predict":
        res = result.get("result", {})
        if res.get("error"):
            return res["error"]
        m = res.get("metric", {})
        if m.get("type") == "classification":
            return f"I trained a {m.get('model')} model to predict '{res.get('target')}' with {m.get('accuracy')} accuracy."
        return (
            f"I trained a {m.get('model')} model to predict '{res.get('target')}' "
            f"with an R² score of {m.get('r2_score')}."
        )

    if rtype == "forecast":
        res = result.get("result", {})
        if res.get("error"):
            return res["error"]
        return (
            f"I forecast {res.get('forecast_periods')} future periods for '{res.get('target')}'. "
            f"The trend is {res.get('trend')} with a projected change of {res.get('projected_change_percent')}%."
        )

    if rtype == "clean":
        reps = result.get("reports", [])
        if reps:
            rep = reps[0]
            actions = rep.get("actions", [])
            if actions:
                return f"I cleaned the data. Actions taken: {'; '.join(actions[:4])}."
            return "Your data was already clean — no changes were needed."
        return "Data cleaning complete."

    if rtype == "insights":
        res = result.get("result", {})
        if res.get("error"):
            return res["error"]
        findings = res.get("findings", [])
        if findings:
            return findings[0]
        return "Here are the smart insights for your data."

    if rtype == "anomalies":
        res = result.get("result", {})
        if res.get("error"):
            return res["error"]
        if res.get("message"):
            return res["message"]
        anoms = res.get("anomalies", {})
        if anoms:
            names = ", ".join(list(anoms.keys())[:3])
            return f"I detected anomalies in: {names}. See details below."
        return "No significant anomalies detected in the numeric columns."

    if rtype == "report":
        return "I've generated your executive report. You can download the PDF below."

    if rtype == "text":
        res = result.get("result", {})
        if res.get("error"):
            return res["error"]
        return (
            f"Your text contains {res.get('word_count')} words, "
            f"{res.get('sentence_count')} sentences, and {res.get('character_count')} characters."
        )

    if rtype == "help":
        return "Here are the available commands you can use."

    return "Here's what I found in your data."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n🚀 Auto Data Analyst Agent running at http://127.0.0.1:{}\n".format(port))
    app.run(debug=True, host="0.0.0.0", port=port)
