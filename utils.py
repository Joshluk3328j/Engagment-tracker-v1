# utils.py

import os
import csv
from datetime import datetime
import math
import numpy as np
import cv2
from fpdf import FPDF  # pip install fpdf2
import matplotlib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from collections import Counter
import re
from reportlab.platypus import PageBreak
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
matplotlib.use('Agg')
apikey = os.environ["GEMINI_API_KEY"]

# Import OUTPUT_DIR from config
from config import OUTPUT_DIR

# ========== UTILITY FUNCTIONS ==========
def timestamp():
    """Returns current time in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

def save_csv(filename, rows, headers):
    """Saves data to a CSV file with given headers."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

# ========== GRAPH GENERATION ==========
def generate_engagement_graphs(events):
    """
    Generates and saves graphs for attention, fatigue, and hand motion.
    """
    attention_data = []
    fatigue_yawns = []
    fatigue_blinks = []
    hand_raised = []
    hand_detected = []

    for ts_str, event_type, description, _ in events:
        dt_obj = datetime.strptime(ts_str, "%H:%M:%S").time()
        dummy_date = datetime(2000, 1, 1)
        full_dt_obj = datetime.combine(dummy_date, dt_obj)

        if event_type == "Attention":
            is_focused = 1 if description == "Focused" else 0
            attention_data.append((full_dt_obj, is_focused))
        elif event_type == "Fatigue":
            if "Yawning" in description:
                fatigue_yawns.append(full_dt_obj)
            elif "Blink" in description:
                fatigue_blinks.append(full_dt_obj)
        elif event_type == "Hand Motion":
            if "Hand Raised" in description:
                hand_raised.append(full_dt_obj)
            elif "Hand Detected" in description:
                hand_detected.append(full_dt_obj)

    # Attention plot
    if attention_data:
        times = [item[0] for item in attention_data]
        states = [item[1] for item in attention_data]
        
        plt.figure(figsize=(12, 6))

        # Highlight distracted periods
        distraction_segments = []
        start_distraction = None
        for current_time, current_state in attention_data:
            if current_state == 0 and start_distraction is None:
                start_distraction = current_time
            elif current_state == 1 and start_distraction is not None:
                distraction_segments.append((start_distraction, current_time))
                start_distraction = None
        if start_distraction is not None:
            distraction_segments.append((start_distraction, times[-1]))

        for start, end in distraction_segments:
            plt.axvspan(start, end, color='red', alpha=0.2,
                        label='Distracted' if start == distraction_segments[0][0] else "")

        plt.title('Student Distraction Periods Over Time')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.yticks([0, 1], ['Distracted', 'Focused'])  # Keep y-ticks for context
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "attention_over_time.png"))
        plt.close()

    # Fatigue plot
    if fatigue_yawns or fatigue_blinks:
        plt.figure(figsize=(12, 4))
        if fatigue_yawns:
            plt.scatter(fatigue_yawns, [1] * len(fatigue_yawns),
                        color='orange', marker='o', label='Yawn Detected')
        if fatigue_blinks:
            plt.scatter(fatigue_blinks, [0] * len(fatigue_blinks),
                        color='purple', marker='x', label='Blink Fatigue')
        plt.title('Fatigue Events Over Time')
        plt.xlabel('Time')
        plt.ylabel('Event Type')
        plt.yticks([0, 1], ['Blink Fatigue', 'Yawn Detected'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fatigue_events.png"))
        plt.close()

    # Hand motion plot
    if hand_raised or hand_detected:
        plt.figure(figsize=(12, 4))
        if hand_raised:
            plt.scatter(hand_raised, [1] * len(hand_raised),
                        color='blue', marker='^', label='Hand Raised')
        if hand_detected:
            plt.scatter(hand_detected, [0] * len(hand_detected),
                        color='cyan', marker='s', label='Hand Detected (Frequent)')
        plt.title('Hand Motion Events Over Time')
        plt.xlabel('Time')
        plt.ylabel('Event Type')
        plt.yticks([0, 1], ['Hand Detected', 'Hand Raised'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "hand_motion_events.png"))
        plt.close()

# ========== PDF REPORT ==========
class PDFReport(FPDF):
    def header(self):
        if self.page_no() != 1:
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 10, "AI Interview Engagement Report", ln=True, align="C")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def summarize_transcription(transcriptions):
    """Simple extractive summarizer."""
    if not transcriptions:
        return "No speech was transcribed."

    # Combine all transcript text
    text = " ".join([text for _, text in transcriptions])
    prompt_template = PromptTemplate(
        input_variables=["text"], 
        template="""
        forget the timestamps in the meeting text below and summarize the text stating what the meeting was about:
        {text}
        
        Summary: 
        """
    )
    llms= ChatGoogleGenerativeAI( model="gemini-1.5-flash",
        google_api_key=apikey,
        temperature=0.5,
        max_retries=3)
    
    chain = prompt_template | llms
    
    response = chain.invoke({"text":text})
    
    return response.content


def generate_full_report(transcriptions, events, output_dir):
    pdf_path = os.path.join(output_dir, "engagement_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("EngageTrack Session Report", styles['Title']))
    elements.append(Spacer(1, 20))

    # Summary statistics
    total_events = len(events)
    total_transcripts = len(transcriptions)
    elements.append(Paragraph(f"Total Events Logged: {total_events}", styles['Normal']))
    elements.append(Paragraph(f"Total Transcriptions: {total_transcripts}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Event Table
    elements.append(Paragraph("Event Log", styles['Heading2']))
    table_data = [["Timestamp", "Event Type", "Description", "Speech Context"]]
    max_width = 200
    for ts, event_type, desc, speech_ctx in events:
        # Wrap the speech context text using Paragraph
        speech_ctx = speech_ctx if speech_ctx else ""
        wrapped_speech_ctx = Paragraph(speech_ctx, styles['Normal'])  # Wrap text to fit within max_width
        wrapped_desc = Paragraph(desc, styles['Normal'])  # Wrap text to fit within max_width
        table_data.append([ts, event_type, wrapped_desc, wrapped_speech_ctx])
    
    col_widths = [80, 70, 120, max_width]
    table = Table(table_data,colWidths=col_widths,repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # --- Summary Section ---
    elements.append(Paragraph("Session Summary", styles['Heading2']))
    summary_text = summarize_transcription(transcriptions)
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Transcription Table
    elements.append(Paragraph("Speech Transcription", styles['Heading2']))
    for ts, text in transcriptions:
        elements.append(Paragraph(f"[{ts}] {text}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # --- Start graphs on a new page ---
    graphs_dir = output_dir
    graph_files = [f for f in os.listdir(graphs_dir) if f.endswith(".png")]

    if graph_files:
        elements.append(PageBreak())
        elements.append(Paragraph("Graphs and Visualizations", styles['Title']))
        elements.append(Spacer(1, 20))

        for file in graph_files:
            title = os.path.splitext(file)[0].replace("_", " ").title()
            elements.append(Paragraph(title, styles['Heading3']))
            elements.append(RLImage(os.path.join(graphs_dir, file), width=400, height=250))
            elements.append(Spacer(1, 20))

    doc.build(elements)
    print(f"[INFO] Full report saved to {pdf_path}")


# ========== FACIAL LANDMARK UTILITIES ==========
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_eye_aspect_ratio(eye_landmarks):
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def get_mouth_aspect_ratio(mouth_landmarks):
    A = euclidean_distance(mouth_landmarks[1], mouth_landmarks[5])
    B = euclidean_distance(mouth_landmarks[2], mouth_landmarks[4])
    C = euclidean_distance(mouth_landmarks[0], mouth_landmarks[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def get_head_pose(face_landmarks, image_shape):
    img_h, img_w, _ = image_shape
    image_points = np.array([
        (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h),
        (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h),
        (face_landmarks[61].x * img_w, face_landmarks[61].y * img_h),
        (face_landmarks[291].x * img_w, face_landmarks[291].y * img_h)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]  # Pitch, Yaw, Roll
