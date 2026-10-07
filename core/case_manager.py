"""
Enterprise Case Management & Evidentiary Master Dossier Engine
Manages multi-exhibit forensic cases, evidence chains of custody, and multi-page judicial audit certificates.
"""

from typing import Dict, Any, List, Optional
import json
import datetime
import uuid
from pathlib import Path
import io

from core.report_generator import ForensicReportGenerator, REPORTLAB_AVAILABLE
if REPORTLAB_AVAILABLE:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class CaseManager:
    """
    Maintains persistent investigative cases and exports comprehensive multi-exhibit legal dossiers.
    """

    def __init__(self, storage_dir: str = "data/cases"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_case(self, title: str, investigator: str, organization: str) -> Dict[str, Any]:
        case_id = f"CASE-{datetime.datetime.utcnow().strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}"
        case_data = {
            "case_id": case_id,
            "title": title,
            "investigator": investigator,
            "organization": organization,
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "status": "OPEN_INVESTIGATION",
            "exhibits": []
        }
        self.save_case(case_data)
        return case_data

    def save_case(self, case_data: Dict[str, Any]):
        case_file = self.storage_dir / f"{case_data['case_id']}.json"
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(case_data, f, indent=4)

    def add_exhibit(self, case_id: str, exhibit_data: Dict[str, Any]) -> Dict[str, Any]:
        case_file = self.storage_dir / f"{case_id}.json"
        if not case_file.exists():
            return None
        with open(case_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)
        
        exhibit_data["exhibit_id"] = f"EXHIBIT-{len(case_data['exhibits']) + 1:02d}"
        exhibit_data["attached_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
        case_data["exhibits"].append(exhibit_data)
        self.save_case(case_data)
        return case_data

    def list_cases(self) -> List[Dict[str, Any]]:
        cases = []
        for p in self.storage_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cases.append(json.load(f))
            except Exception:
                continue
        return sorted(cases, key=lambda x: x.get("created_utc", ""), reverse=True)

    def generate_case_master_pdf(self, case_id: str) -> bytes:
        """Export comprehensive multi-page master judicial forensic certificate."""
        case_file = self.storage_dir / f"{case_id}.json"
        if not case_file.exists():
            return b""
        with open(case_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        if not REPORTLAB_AVAILABLE:
            return json.dumps(case_data, indent=4).encode('utf-8')

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'))
        sub_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'))
        h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1e293b'), spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor('#334155'), leading=12)

        story = []

        # Cover Banner
        story.append(Paragraph("JUDICIAL MASTER FORENSIC EVIDENCE DOSSIER", title_style))
        story.append(Paragraph(f"VERITAS SOTA Cyber-Forensics Lab • Case ID: {case_data['case_id']}", sub_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0284c7'), spaceAfter=14))

        # Case Metadata Table
        meta_table_data = [
            [Paragraph("<b>Investigation Title:</b>", body_style), Paragraph(case_data['title'], body_style)],
            [Paragraph("<b>Lead Investigator:</b>", body_style), Paragraph(case_data['investigator'], body_style)],
            [Paragraph("<b>Commissioning Agency:</b>", body_style), Paragraph(case_data['organization'], body_style)],
            [Paragraph("<b>Date Opened:</b>", body_style), Paragraph(case_data['created_utc'], body_style)],
            [Paragraph("<b>Total Exhibits Attached:</b>", body_style), Paragraph(f"<b>{len(case_data['exhibits'])} Evidence Items</b>", body_style)]
        ]
        t = Table(meta_table_data, colWidths=[140, 400])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Exhibits Inventory Table
        story.append(Paragraph("Evidence Exhibits Inventory & Threat Assessment", h2_style))
        if case_data['exhibits']:
            rows = [["ID", "Media Item", "SHA-256 Hash", "Primary Verdict", "Confidence"]]
            for ex in case_data['exhibits']:
                h_short = (ex.get("sha256", "N/A")[:12] + "...") if ex.get("sha256") else "N/A"
                rows.append([
                    ex.get("exhibit_id", "-"),
                    ex.get("media_name", "Exhibit"),
                    h_short,
                    ex.get("verdict", "Unknown"),
                    f"{ex.get('confidence', 0):.1f}%"
                ])
            ex_table = Table(rows, colWidths=[65, 120, 110, 165, 80])
            ex_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(ex_table)
        else:
            story.append(Paragraph("<i>No exhibits attached to this case.</i>", body_style))

        story.append(Spacer(1, 20))

        # Judicial Statement & Signature Blocks
        story.append(Paragraph("Evidentiary Certification & Sworn Declaration", h2_style))
        declaration = (
            "I hereby declare under penalty of perjury that the multi-cue computational forensic tests "
            "documented herein were conducted using deterministic spatial, spectral, and error-level "
            "forensic algorithms. The cryptographic hashes represent the immutable binary integrity of the exhibits."
        )
        story.append(Paragraph(declaration, body_style))
        story.append(Spacer(1, 25))

        # Signatures
        sig_data = [
            [
                Paragraph("____________________________________________<br/><b>Lead Forensic Examiner Signature</b>", body_style),
                Paragraph("____________________________________________<br/><b>Supervising Evidentiary Officer</b>", body_style)
            ],
            [
                Paragraph(f"Date: {datetime.date.today().isoformat()}", body_style),
                Paragraph(f"Date: {datetime.date.today().isoformat()}", body_style)
            ]
        ]
        sig_table = Table(sig_data, colWidths=[270, 270])
        sig_table.setStyle(TableStyle([
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(sig_table)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
