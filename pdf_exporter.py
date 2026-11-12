import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

import config

class NumberedCanvas(canvas.Canvas):
    """Custom canvas that computes total page count dynamically for footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#6b7280"))
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 20, footer_text)
        self.drawString(36, 20, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.restoreState()


def _clean_records(records, min_duration):
    """Drops invalid rows and duplicates (by session_uid, else by content)."""
    seen, clean = set(), []
    for rec in records:
        try:
            dur = float(rec.get("duration_seconds", 0))
        except (TypeError, ValueError):
            continue
        if dur < min_duration or not rec.get("object_name"):
            continue
        key = rec.get("session_uid") or (
            rec.get("run_id"), rec.get("object_name"), rec.get("tracking_id"),
            rec.get("date"), rec.get("first_seen"), rec.get("last_seen"))
        if key in seen:
            continue
        seen.add(key)
        clean.append(rec)
    clean.sort(key=lambda r: (r.get("date", ""), r.get("first_seen", ""), r.get("tracking_id", 0)))
    return clean


def _fmt_duration(sec):
    sec = float(sec)
    return f"{int(sec // 60)}m {sec % 60:04.1f}s" if sec >= 60 else f"{sec:.1f}s"


def generate_objects_pdf(db, filename_prefix="results/detected_objects_report", limit=500, run_id=None):
    """Generates the PDF of confirmed detection sessions.

    With run_id (main.py passes the current camera run) only that run's sessions
    are listed -- never earlier runs, whose tracker ids restart at 1. Without
    run_id it falls back to today's records. Duplicates and sub-minimum
    durations are removed.
    """
    os.makedirs("results", exist_ok=True)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if run_id:
        raw = db.search_detections(run_id=run_id, limit=limit)
    else:
        raw = db.search_detections(date=today_str, limit=limit)
    records = _clean_records(raw, config.MIN_SESSION_DURATION_SECONDS)
    if not records:
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=45)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=16,
                                 textColor=colors.HexColor('#1f2937'), spaceAfter=4)
    subtitle_style = ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontSize=9,
                                    textColor=colors.HexColor('#4b5563'), spaceAfter=4)
    note_style = ParagraphStyle('ReportNote', parent=subtitle_style, fontSize=7.5, spaceAfter=12)
    cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontSize=8,
                                textColor=colors.HexColor('#1f2937'))
    header_cell_style = ParagraphStyle('TableHeaderCell', parent=styles['Normal'], fontSize=8.5,
                                       fontName="Helvetica-Bold", textColor=colors.whitesmoke, alignment=1)

    scope = f"Camera run {run_id}" if run_id else f"All records for {today_str}"
    elements.append(Paragraph(f"Detected Objects Session Report ({today_str})", title_style))
    elements.append(Paragraph(
        f"{scope} &nbsp;|&nbsp; Confirmed sessions: {len(records)} &nbsp;|&nbsp; "
        f"Distinct objects: {len({r['object_name'] for r in records})} &nbsp;|&nbsp; "
        f"Distinct tracking IDs: {len({r.get('tracking_id') for r in records})}", subtitle_style))
    elements.append(Paragraph(
        "Tracking IDs identify tracks, not people: each detected object (a person, their face, "
        "clothing, glasses...) has its own ID. Only sessions confirmed over several frames are listed.",
        note_style))

    heads = ["#", "Object", "Track ID", "First Seen", "Last Seen", "Duration", "Avg Conf", "Max Conf"]
    table_data = [[Paragraph(h, header_cell_style) for h in heads]]
    for i, rec in enumerate(records, 1):
        table_data.append([
            Paragraph(str(i), cell_style),
            Paragraph(str(rec.get("object_name", "N/A")), cell_style),
            Paragraph(str(rec.get("tracking_id", "N/A")), cell_style),
            Paragraph(str(rec.get("first_seen", "N/A")), cell_style),
            Paragraph(str(rec.get("last_seen", "N/A")), cell_style),
            Paragraph(_fmt_duration(rec.get("duration_seconds", 0)), cell_style),
            Paragraph(f"{float(rec.get('avg_confidence', rec.get('max_confidence', 0))) * 100:.1f}%", cell_style),
            Paragraph(f"{float(rec.get('max_confidence', 0)) * 100:.1f}%", cell_style),
        ])

    t = Table(table_data, colWidths=[26, 115, 52, 70, 70, 74, 62, 62], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f9fafb')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
    ]))
    elements.append(t)
    doc.build(elements, canvasmaker=NumberedCanvas)
    return filename
