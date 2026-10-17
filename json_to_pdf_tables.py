#!/usr/bin/env python3
"""
Convert circuit analytics JSON to formatted PDF tables
"""

import json
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def load_json(filename):
    """Load JSON results"""
    with open(filename, 'r') as f:
        return json.load(f)


def create_pdf_tables(json_data, output_filename):
    """Create PDF with tables for each case and backend"""
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=landscape(letter),
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch
    )

    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    case_style = ParagraphStyle(
        'CaseTitle',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#2e5090'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    # Main title
    story.append(Paragraph("Circuit Analytics Results", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Iterate through each case
    for case_name, case_data in json_data.items():
        story.append(Paragraph(f"{case_name}", case_style))

        for backend_name, backend_data in case_data.items():
            # Check if backend was skipped due to error
            if "error" in backend_data:
                story.append(Paragraph(f"<b>{backend_name}</b>: {backend_data['error']}", styles['Normal']))
                story.append(Spacer(1, 0.15*inch))
                continue

            # Create table for this backend
            table_data = [['Circuit Type', 'Opt Level', '2Q Gates', '2Q Gate Depth', 'Total Depth']]

            for circ_type, circ_data in backend_data.items():
                for opt_level, analytics in circ_data.items():
                    # Skip error entries
                    if "error" in analytics:
                        continue
                    opt_num = opt_level.split('_')[2]
                    table_data.append([
                        circ_type.replace('_', ' ').title(),
                        f"L{opt_num}",
                        str(analytics['total_2q_gates']),
                        str(analytics['two_qubit_gate_depth']),
                        str(analytics['depth'])
                    ])

            # Only add table if there's data (more than just header)
            if len(table_data) > 1:
                # Create table
                table = Table(table_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.2*inch, 1.2*inch])
                table.setStyle(TableStyle([
                    # Header row
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('TOPPADDING', (0, 0), (-1, 0), 8),
                    # Data rows
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('TOPPADDING', (0, 1), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ]))

                # Add table with backend name
                story.append(Paragraph(f"<b>{backend_name}</b>", styles['Normal']))
                story.append(table)
                story.append(Spacer(1, 0.25*inch))

        # Page break after each case (except the last)
        if case_name != list(json_data.keys())[-1]:
            story.append(PageBreak())

    # Build PDF
    doc.build(story)
    print(f"PDF generated: {output_filename}")


def create_summary_csv(json_data, output_filename):
    """Also create a CSV summary for easy reference"""
    rows = []

    for case_name, case_data in json_data.items():
        for backend_name, backend_data in case_data.items():
            # Skip backends with errors
            if "error" in backend_data:
                continue

            for circ_type, circ_data in backend_data.items():
                for opt_level, analytics in circ_data.items():
                    # Skip error entries
                    if "error" in analytics:
                        continue
                    opt_num = opt_level.split('_')[2]
                    rows.append({
                        'Case': case_name,
                        'Backend': backend_name,
                        'Circuit Type': circ_type,
                        'Opt Level': opt_num,
                        '2Q Gates': analytics['total_2q_gates'],
                        '2Q Gate Depth': analytics['two_qubit_gate_depth'],
                        'Total Depth': analytics['depth']
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False)
    print(f"CSV summary generated: {output_filename}")


def main():
    json_file = 'circuit_analytics_results.json'
    pdf_file = 'circuit_analytics_results.pdf'
    csv_file = 'circuit_analytics_summary.csv'

    print(f"Loading {json_file}...")
    data = load_json(json_file)

    print(f"Generating {pdf_file}...")
    create_pdf_tables(data, pdf_file)

    print(f"Generating {csv_file}...")
    create_summary_csv(data, csv_file)

    print("\nDone!")


if __name__ == "__main__":
    main()
