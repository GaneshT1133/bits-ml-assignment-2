"""Generate the final M.Tech Machine Learning Assignment 2 submission PDF."""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = ROOT / "BITS_ML_Assignment_2_Submission.pdf"
REPOSITORY_URL = "https://github.com/GaneshT1133/bits-ml-assignment-2"
APP_URL = "https://bits-ml-assignment-2-32br9urkmmonneej96cxbn.streamlit.app/"

METRIC_HEADERS = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_ROWS = [
    ["Logistic Regression", "0.7438", "0.8239", "0.7730", "0.7368", "0.7545", "0.4874"],
    ["Decision Tree", "0.7531", "0.7509", "0.7614", "0.7836", "0.7723", "0.5031"],
    ["KNN", "0.7438", "0.8144", "0.7697", "0.7427", "0.7560", "0.4867"],
    ["Naive Bayes", "0.7125", "0.7733", "0.7484", "0.6959", "0.7212", "0.4264"],
    ["Support Vector Machine", "0.7406", "0.8373", "0.7857", "0.7076", "0.7446", "0.4853"],
]
OBSERVATION_ROWS = [
    (
        "Logistic Regression",
        "Reached 74.38% accuracy and 0.8239 AUC, with a balanced precision/recall profile "
        "(0.7730 / 0.7368). It provides a strong, interpretable linear baseline.",
    ),
    (
        "Decision Tree",
        "Had the highest accuracy (75.31%), recall (78.36%), F1 (0.7723), and MCC (0.5031) "
        "in this split. Its AUC (0.7509) is the lowest, so its threshold ranking is weaker "
        "despite its best threshold-based scores.",
    ),
    (
        "KNN",
        "Matched Logistic Regression's 74.38% accuracy and achieved 0.8144 AUC. Precision "
        "and recall are close (0.7697 / 0.7427), resulting in a 0.7560 F1 score.",
    ),
    (
        "Naive Bayes",
        "Scored 71.25% accuracy and 0.7212 F1, the lowest on both measures in this "
        "comparison. It is the weakest overall model on this split, while providing a "
        "useful probabilistic baseline.",
    ),
    (
        "Support Vector Machine",
        "Achieved the best AUC (0.8373) and precision (0.7857), indicating the strongest "
        "probability ranking and positive-prediction precision here. Its recall is 0.7076 "
        "and accuracy is 74.06%.",
    ),
    (
        "Overall Winner",
        "Decision Tree is the winner when prioritizing accuracy, recall, F1, and MCC. "
        "Support Vector Machine has the best AUC and precision. The preferred model "
        "depends on the metric priority; these results reflect one fixed stratified split "
        "and do not guarantee generalization.",
    ),
]


def make_styles():
    """Create a consistent typography palette for the PDF."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SubmissionTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#18382f"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubmissionSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64746d"),
            alignment=TA_CENTER,
            spaceAfter=15,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#8c3047"),
            spaceBefore=8,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyTextCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#27352f"),
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "link": ParagraphStyle(
            "LinkText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=15,
            textColor=colors.HexColor("#27352f"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=colors.HexColor("#27352f"),
            alignment=TA_CENTER,
        ),
        "table_model": ParagraphStyle(
            "TableModel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.7,
            leading=9.5,
            textColor=colors.HexColor("#27352f"),
            alignment=TA_LEFT,
        ),
        "observation": ParagraphStyle(
            "ObservationCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#27352f"),
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "SmallNote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64746d"),
            spaceBefore=7,
        ),
    }


def add_page_chrome(canvas, doc):
    """Draw understated page header/footer with page count."""
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(colors.HexColor("#dce5de"))
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, page_height - 15 * mm, page_width - doc.rightMargin, page_height - 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64746d"))
    canvas.drawString(doc.leftMargin, page_height - 12 * mm, "M.Tech · Machine Learning Assignment 2")
    canvas.line(doc.leftMargin, 14 * mm, page_width - doc.rightMargin, 14 * mm)
    canvas.drawRightString(page_width - doc.rightMargin, 9.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_story(styles):
    """Compose the requested three included sections in their requested order."""
    story = [
        Spacer(1, 5 * mm),
        Paragraph("BITS Machine Learning Assignment 2", styles["title"]),
        Paragraph("M.Tech · AIML · Final Submission", styles["subtitle"]),
        Paragraph("1. Mandatory Submission Links", styles["section"]),
        Paragraph(
            f"<b>GitHub Repository:</b> "
            f"<link href='{REPOSITORY_URL}' color='#356b56'>{escape(REPOSITORY_URL)}</link>",
            styles["link"],
        ),
        Paragraph(
            f"<b>Live Streamlit App:</b> "
            f"<link href='{APP_URL}' color='#356b56'>{escape(APP_URL)}</link>",
            styles["link"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("3. GitHub README Content (Part A &amp; B)", styles["section"]),
        Paragraph("Problem Statement", styles["body"]),
        Paragraph(
            "Building an end-to-end machine-learning classification pipeline and interactive "
            "web deployment using the UCI Wine Quality dataset. The project trains and "
            "compares five classifiers to predict a binary wine-quality label and provides "
            "an interactive Streamlit app for evaluation on labeled test data.",
            styles["body"],
        ),
        Paragraph("Dataset Description", styles["body"]),
        Paragraph(
            "The UCI Red Wine Quality dataset contains 1,599 instances. It includes 11 "
            "original physicochemical measurements and three engineered features: "
            "<font name='Courier'>total_acidity</font>, "
            "<font name='Courier'>bound_sulfur_dioxide</font>, and "
            "<font name='Courier'>sugar_to_acidity_ratio</font>. These form 14 input "
            "features in total. The binary target, "
            "<font name='Courier'>quality_label</font>, is 1 when the original quality "
            "score is at least 6 and 0 otherwise; the original quality column is dropped.",
            styles["body"],
        ),
        Spacer(1, 3 * mm),
        Paragraph("4. Metrics Comparison Table &amp; Observations Table", styles["section"]),
        Paragraph(
            "All scores below were computed on the same stratified 320-instance test "
            "partition. AUC is based on predicted class probabilities.",
            styles["body"],
        ),
    ]

    metric_data = [[Paragraph(escape(header), styles["table_header"]) for header in METRIC_HEADERS]]
    for row in METRIC_ROWS:
        metric_data.append(
            [Paragraph(escape(row[0]), styles["table_model"])]
            + [Paragraph(escape(value), styles["table_cell"]) for value in row[1:]]
        )

    metric_table = Table(
        metric_data,
        colWidths=[119, 56, 49, 65, 53, 47, 53],
        repeatRows=1,
        hAlign="LEFT",
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24483b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cbd7ce")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f1")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            metric_table,
            PageBreak(),
            Paragraph("Observations by Model", styles["body"]),
        ]
    )

    observation_data = [
        [Paragraph("ML Model Name", styles["table_header"]), Paragraph("Observation about model performance", styles["table_header"])]
    ]
    for model_name, observation in OBSERVATION_ROWS:
        observation_data.append(
            [
                Paragraph(escape(model_name), styles["table_model"]),
                Paragraph(escape(observation), styles["observation"]),
            ]
        )

    observation_table = Table(
        observation_data,
        colWidths=[119, 323],
        repeatRows=1,
        hAlign="LEFT",
    )
    observation_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8c3047")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#cbd7ce")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f4f3")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            observation_table,
            Paragraph(
                "Winner summary: Decision Tree leads on accuracy, recall, F1, and MCC; "
                "Support Vector Machine leads on AUC and precision. Model choice depends "
                "on the metric priority.",
                styles["small"],
            ),
        ]
    )
    return story


def main():
    """Write the PDF submission beside this script."""
    styles = make_styles()
    document = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="BITS M.Tech Machine Learning Assignment 2 Submission",
        author="M.Tech AIML Student",
        subject="Wine Quality classification project submission",
    )
    document.build(build_story(styles), onFirstPage=add_page_chrome, onLaterPages=add_page_chrome)
    print(f"Created PDF submission: {OUTPUT_PDF}")
    print(f"File size: {OUTPUT_PDF.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
