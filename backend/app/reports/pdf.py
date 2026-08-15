from __future__ import annotations
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from .schemas import Report

class PdfReportGenerator:
    def generate(self, report: Report) -> bytes:
        buffer = BytesIO(); document = SimpleDocTemplate(buffer, pagesize=letter); styles = getSampleStyleSheet(); flow = [Paragraph(report.title, styles["Title"])]
        for heading, content in (("Executive Summary", report.executive_summary), ("Dataset Overview", report.dataset_overview), ("Data Quality", report.data_quality), ("Key Performance Indicators", report.kpis), ("Key Insights", [i.get("description", i.get("title", "")) for i in report.insights]), ("Recommendations", report.recommendations), ("Forecast", report.forecast), ("Methodology", report.methodology)):
            flow.extend([Spacer(1, 10), Paragraph(heading, styles["Heading2"]), Paragraph(self._text(content), styles["BodyText"])])
        if report.kpis:
            table = Table([["KPI", "Value"]] + [[item["name"], str(item["value"])] for item in report.kpis]); table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), .25, colors.grey)])); flow.append(table)
        document.build(flow); return buffer.getvalue()
    @staticmethod
    def _text(value: object) -> str: return str(value).replace("\n", "<br/>")
