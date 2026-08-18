from __future__ import annotations
from .builder import ReportBuilder
from .excel import ExcelReportGenerator
from .pdf import PdfReportGenerator
from .powerpoint import PowerPointReportGenerator

class ReportEngine:
    def generate(self, dataset_id: str, analysis: dict, output_format: str) -> tuple[object, bytes, str]:
        report = ReportBuilder().build(dataset_id, analysis); generators = {"pdf": (PdfReportGenerator(), "application/pdf"), "excel": (ExcelReportGenerator(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"), "powerpoint": (PowerPointReportGenerator(), "application/vnd.openxmlformats-officedocument.presentationml.presentation")}
        if output_format not in generators: raise ValueError("Unsupported format. Use pdf, excel, or powerpoint.")
        generator, content_type = generators[output_format]
        return report, generator.generate(report), content_type
