"""
ReportService — Generates structured CSV and PDF compliance & attendance reports for SiteSentinel.
"""
import io
import csv
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        start_page_fn = getattr(self, "_startPage", None)
        if callable(start_page_fn):
            start_page_fn()
        else:
            super().showPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header
        self.drawString(40, 755, "SiteSentinel — AI Safety & Attendance Compliance Audit Report")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 748, 572, 748)
        
        # Footer
        self.line(40, 45, 572, 45)
        self.drawString(40, 32, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Confidential")
        current_page = getattr(self, "_pageNumber", 1)
        self.drawRightString(572, 32, f"Page {current_page} of {page_count}")
        self.restoreState()


def generate_csv_report(manager_name: str, workers: list, attendance_logs: list, safety_events: list, from_date: str = "", to_date: str = "") -> str:
    """Generates a comprehensive CSV report containing summary, attendance, and safety violation logs."""
    output = io.StringIO()
    writer = csv.writer(output)

    # 1. Summary Header
    writer.writerow(["SITESENTINEL COMPLIANCE & ATTENDANCE AUDIT REPORT"])
    writer.writerow(["Manager / Account", manager_name or "Site Manager"])
    writer.writerow(["Report Date Range", f"{from_date or 'Beginning'} to {to_date or 'Present'}"])
    writer.writerow(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Total Registered Workers", len(workers)])
    writer.writerow(["Total Attendance Logs", len(attendance_logs)])
    writer.writerow(["Total Safety Events", len(safety_events)])
    writer.writerow([])

    # 2. Worker Roster & Latest Status
    writer.writerow(["── WORKER STATUS ROSTER ──"])
    writer.writerow(["Worker DB ID", "Worker ID", "Name", "Assigned Helmet", "Status", "Check-in Time", "Checkout Time", "Latest PPE Score"])
    for w in workers:
        writer.writerow([
            w.get("id", ""),
            w.get("worker_id", ""),
            f"{w.get('first_name', '')} {w.get('last_name', '')}".strip(),
            w.get("helmet_id", "") or "None",
            w.get("status", "Off-Site"),
            w.get("checkin_time", "--:--"),
            w.get("checkout_time", "--:--"),
            f"{w.get('ppe_score', 0)}%" if w.get('ppe_score') is not None else "N/A"
        ])
    writer.writerow([])

    # 3. Attendance Logs
    writer.writerow(["── ATTENDANCE AUDIT TRAIL ──"])
    writer.writerow(["Log ID", "Date", "Timestamp", "Worker ID", "Event", "PPE Score", "Helmet ID"])
    for log in attendance_logs:
        writer.writerow([
            log.get("id", ""),
            log.get("date", ""),
            log.get("timestamp", ""),
            log.get("worker_id", ""),
            log.get("event", ""),
            f"{log.get('ppe_score', 0)}%",
            log.get("helmet_id", "") or "None"
        ])
    writer.writerow([])

    # 4. Safety Events & Violations
    writer.writerow(["── SAFETY & PPE VIOLATIONS AUDIT ──"])
    writer.writerow(["Event ID", "Timestamp", "Worker ID", "Event Type", "Message", "PPE Score", "Helmet ID", "Acknowledged"])
    for ev in safety_events:
        writer.writerow([
            ev.get("id", ""),
            ev.get("created_at", ""),
            ev.get("worker_id", "") or "N/A",
            ev.get("event_type", ""),
            ev.get("message", ""),
            f"{ev.get('ppe_score', '')}%" if ev.get('ppe_score') is not None else "N/A",
            ev.get("helmet_id", "") or "N/A",
            "Yes" if ev.get("acknowledged") in [1, "1", True] else "No"
        ])

    return output.getvalue()


def generate_pdf_report(manager_name: str, workers: list, attendance_logs: list, safety_events: list, from_date: str = "", to_date: str = "") -> bytes:
    """Generates a professional multi-page PDF audit report with executive summaries and tables."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B')
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=14,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#334155')
    )
    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0F172A')
    )
    cell_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    elements = []

    # Title Block
    elements.append(Paragraph("SiteSentinel", ParagraphStyle('Brand', fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=colors.HexColor('#FF6B2B'))))
    elements.append(Paragraph("AI-Powered Safety & Attendance Compliance Audit Report", title_style))
    date_str = f"Date Range: <b>{from_date or 'Beginning'}</b> to <b>{to_date or 'Present'}</b>  |  Site Manager: <b>{manager_name or 'Authorized Manager'}</b>"
    elements.append(Paragraph(date_str, subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#FF6B2B'), spaceAfter=14))

    # Executive KPI Summary Cards
    total_workers = len(workers)
    active_onsite = sum(1 for w in workers if w.get("status") in ["Active", "On-Site"])
    total_attendance = len(attendance_logs)
    violations_count = sum(1 for ev in safety_events if "violation" in ev.get("event_type", "").lower() or ev.get("event_type") == "PPE_VIOLATION")

    kpi_data = [
        [
            Paragraph(f"<b>Registered Workers</b><br/><font size='14' color='#0F172A'><b>{total_workers}</b></font>", cell_style),
            Paragraph(f"<b>Currently On-Site</b><br/><font size='14' color='#00D4AA'><b>{active_onsite}</b></font>", cell_style),
            Paragraph(f"<b>Attendance Logs</b><br/><font size='14' color='#7C8FF5'><b>{total_attendance}</b></font>", cell_style),
            Paragraph(f"<b>Safety Violations</b><br/><font size='14' color='#FF4757'><b>{violations_count}</b></font>", cell_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 142])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 14))

    # 1. Worker Status Roster Table
    elements.append(Paragraph("1. Worker Status Roster & Safety Ratings", section_heading))
    worker_rows = [[
        Paragraph("Worker ID", cell_header),
        Paragraph("Worker Name", cell_header),
        Paragraph("Assigned Helmet", cell_header),
        Paragraph("Status", cell_header),
        Paragraph("Check-in", cell_header),
        Paragraph("Checkout", cell_header),
        Paragraph("PPE Score", cell_header)
    ]]
    for w in workers[:40]:  # Cap to reasonable list
        score = w.get("ppe_score")
        score_txt = f"{score}%" if score is not None else "N/A"
        worker_rows.append([
            Paragraph(str(w.get("worker_id", "")), cell_bold),
            Paragraph(f"{w.get('first_name', '')} {w.get('last_name', '')}".strip(), cell_style),
            Paragraph(str(w.get("helmet_id") or "—"), cell_style),
            Paragraph(str(w.get("status", "Off-Site")), cell_style),
            Paragraph(str(w.get("checkin_time", "--:--")), cell_style),
            Paragraph(str(w.get("checkout_time", "--:--")), cell_style),
            Paragraph(score_txt, cell_bold)
        ])
    if len(worker_rows) == 1:
        worker_rows.append([Paragraph("No worker records found for this criteria.", cell_style)] * 7)

    worker_table = Table(worker_rows, colWidths=[70, 120, 80, 70, 65, 65, 62])
    worker_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(worker_table)
    elements.append(Spacer(1, 14))

    # 2. Attendance Trail
    elements.append(Paragraph("2. Gate Attendance Trail (Latest Logs)", section_heading))
    att_rows = [[
        Paragraph("Timestamp", cell_header),
        Paragraph("Worker ID", cell_header),
        Paragraph("Event", cell_header),
        Paragraph("PPE Score", cell_header),
        Paragraph("Helmet", cell_header)
    ]]
    for log in attendance_logs[:50]:
        att_rows.append([
            Paragraph(str(log.get("timestamp", log.get("date", ""))), cell_style),
            Paragraph(str(log.get("worker_id", "")), cell_bold),
            Paragraph(str(log.get("event", "")), cell_style),
            Paragraph(f"{log.get('ppe_score', 0)}%", cell_bold),
            Paragraph(str(log.get("helmet_id") or "—"), cell_style)
        ])
    if len(att_rows) == 1:
        att_rows.append([Paragraph("No attendance records found.", cell_style)] * 5)

    att_table = Table(att_rows, colWidths=[140, 90, 100, 90, 112])
    att_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#334155')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(att_table)
    elements.append(Spacer(1, 14))

    # 3. Safety Violations & Incident Log
    elements.append(Paragraph("3. Safety Violations & System Events", section_heading))
    ev_rows = [[
        Paragraph("Date/Time", cell_header),
        Paragraph("Worker", cell_header),
        Paragraph("Type", cell_header),
        Paragraph("Details / Message", cell_header),
        Paragraph("Status", cell_header)
    ]]
    for ev in safety_events[:50]:
        ack = "Acknowledged" if ev.get("acknowledged") in [1, "1", True] else "Unreviewed"
        ev_rows.append([
            Paragraph(str(ev.get("created_at", "")), cell_style),
            Paragraph(str(ev.get("worker_id") or "Gate"), cell_bold),
            Paragraph(str(ev.get("event_type", "")), cell_style),
            Paragraph(str(ev.get("message", "")), cell_style),
            Paragraph(ack, cell_style)
        ])
    if len(ev_rows) == 1:
        ev_rows.append([Paragraph("No safety events logged for this period.", cell_style)] * 5)

    ev_table = Table(ev_rows, colWidths=[120, 70, 90, 162, 90])
    ev_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#475569')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(ev_table)

    # Build PDF with NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
