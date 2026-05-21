import os
import io
import json
import time
from typing import Optional

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

from predict import extract_features
from heatmap import generate_heatmap
from clustering import generate_cluster_plot
from report import create_report


BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")
HISTORY_PATH = os.path.join(BASE_DIR, "static", "history.json")
PATIENTS_PATH = os.path.join(BASE_DIR, "static", "patients.json")
ANALYSES_PATH = os.path.join(BASE_DIR, "static", "analyses.json")
DELETED_PATH = os.path.join(BASE_DIR, "static", "deleted_patients.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


def save_upload(uploaded_file):
    path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def load_image(path):
    try:
        return Image.open(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def extract_features_cached(path):
    return extract_features(path)


def append_history(entry):
    data = []
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                data = json.load(f)
    except Exception:
        data = []

    data.insert(0, entry)
    data = data[:20]

    with open(HISTORY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_patients():
    try:
        if os.path.exists(PATIENTS_PATH):
            with open(PATIENTS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        return []
    return []


def save_patients(patients):
    with open(PATIENTS_PATH, "w") as f:
        json.dump(patients, f, indent=2)


def find_patient_index(patients, patient_id):
    for index, patient in enumerate(patients):
        if patient.get("patient_id") == patient_id:
            return index
    return None


def get_patient(patients, patient_id):
    for patient in patients:
        if patient.get("patient_id") == patient_id:
            return patient
    return None


def load_analyses():
    try:
        if os.path.exists(ANALYSES_PATH):
            with open(ANALYSES_PATH, "r") as f:
                return json.load(f)
    except Exception:
        return []
    return []


def save_analyses(analyses):
    with open(ANALYSES_PATH, "w") as f:
        json.dump(analyses, f, indent=2)


def load_deleted_patients():
    try:
        if os.path.exists(DELETED_PATH):
            with open(DELETED_PATH, "r") as f:
                return json.load(f)
    except Exception:
        return []
    return []


def append_deleted_patient(entry):
    data = []
    try:
        if os.path.exists(DELETED_PATH):
            with open(DELETED_PATH, "r") as f:
                data = json.load(f)
    except Exception:
        data = []

    data.insert(0, entry)
    data = data[:50]

    with open(DELETED_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_patient_recommendations(patient, last_score: float | None = None):
    recs = []
    age = patient.get("age")
    history = patient.get("history", "") or ""
    if last_score is not None:
        if last_score >= 0.75:
            recs.append("High-risk abnormality detected; schedule urgent radiologist review.")
            recs.append("Consider additional imaging such as CT or MRI for further investigation.")
            recs.append("If available, compare with prior scans to identify progression.")
        elif last_score >= 0.50:
            recs.append("Moderate abnormality detected; review findings with a radiologist.")
            recs.append("Recommend follow-up imaging in 4-6 weeks or specialist consultation.")
        else:
            recs.append("Low abnormality signal; maintain routine clinical monitoring.")
            recs.append("Recommend annual follow-up imaging or symptom-based review.")
    else:
        recs.append("No scan available yet; register a patient scan to generate tailored recommendations.")

    if age is not None and int(age) >= 60:
        recs.append("Elderly patient — prioritize careful review and consider age-related risk factors.")

    if "diabetes" in history.lower():
        recs.append("History of diabetes: coordinate management with endocrinology and monitor for infection risk.")
    if "smoke" in history.lower() or "smoking" in history.lower():
        recs.append("Smoking history noted: assess for chronic lung disease and counsel smoking cessation.")
    if "cough" in history.lower() or "fever" in history.lower():
        recs.append("Respiratory symptoms present: evaluate for infection as part of the imaging review.")

    if not recs:
        recs.append("Keep patient under routine monitoring and update the file with any new symptoms.")
    return recs


def read_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
    except Exception:
        return []
    return []


def overlay_images(original_path, heatmap_path, alpha=0.5):
    try:
        orig = Image.open(original_path).convert("RGBA")
        hm = Image.open(heatmap_path).convert("RGBA")
        hm = hm.resize(orig.size)
        blended = Image.blend(orig, hm, alpha=alpha)
        return blended
    except Exception:
        return None


def main():
    st.set_page_config(page_title="Chest X-ray Intelligence Hub", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        color: #0f172a;
    }
    .css-1d391kg, .css-1v3fvcr {
        background: #111827 !important;
    }
    .css-1d391kg .css-1hynsf3, .css-1v3fvcr h2, .css-1v3fvcr h3, .css-1v3fvcr p {
        color: #f8fafc !important;
    }
    .card { background: rgba(255,255,255,0.98); border-radius: 24px; padding: 28px; box-shadow: 0px 24px 48px rgba(15,23,42,0.12); margin-bottom: 28px; }
    .hero { display: flex; flex-wrap: wrap; gap: 24px; align-items: center; justify-content: space-between; margin-bottom: 22px; }
    .hero-text { max-width: 760px; }
    .hero-text h1 { font-size: 48px; margin: 0; line-height: 1.04; letter-spacing: -0.03em; }
    .hero-text p { color: #475569; font-size: 18px; margin: 14px 0 0; }
    .hero-badge { background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%); color: white; padding: 14px 20px; border-radius: 999px; display: inline-block; font-weight: 700; margin-bottom: 18px; }
    .metric-block { border: 1px solid #e2e8f0; padding: 20px; border-radius: 22px; background: #ffffff; }
    .metric-block h4 { margin: 0 0 8px; color: #0f172a; }
    .metric-block span { font-size: 32px; font-weight: 800; color: #1f2937; }
    .section-title { margin-top: 42px; margin-bottom: 22px; }
    .about-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 24px; align-items: stretch; }
    .about-card { background: #ffffff; padding: 26px; border-radius: 26px; box-shadow: 0 22px 60px rgba(15, 23, 42, 0.08); border: 1px solid rgba(226, 232, 240, 0.95); }
    .about-card h3 { margin-top: 0; }
    .about-badge { display: inline-block; background: linear-gradient(90deg, #2563eb 0%, #38bdf8 100%); color: white; padding: 10px 18px; border-radius: 999px; font-weight: 700; margin-bottom: 16px; }
    .animated-icon { width: 100%; max-width: 320px; margin: 0 auto 24px; display: block; }
    .pulse-circle { animation: pulse 2.4s infinite ease-in-out; transform-origin: center; }
    @keyframes pulse {
      0% { transform: scale(1); opacity: 0.82; }
      50% { transform: scale(1.06); opacity: 1; }
      100% { transform: scale(1); opacity: 0.82; }
    }
    .custom-button button, .stButton>button, .stDownloadButton button { background-color: #2563eb !important; color: white !important; border-radius: 14px !important; border: none !important; padding: 10px 16px !important; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div>div {
        border-radius: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class='hero'>
            <div class='hero-text'>
                <div class='hero-badge'>Clinical Workflow</div>
                <h1>Chest X-ray Intelligence Hub</h1>
                <p>Enterprise-style patient management, scan analysis, and reporting for diagnostic teams. Streamline clinical review with AI-informed insights and secure document delivery.</p>
            </div>
            <div class='card' style='max-width: 380px;'>
                <h4>Operational readiness</h4>
                <p>Register patients, analyze X-ray scans, generate PDF reports, and access patient-specific clinical guidance — all from a single intelligent dashboard.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.header("Workspace")
    st.sidebar.markdown("Manage patient intake, imaging review, and reporting with a modern clinical workflow.")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Patients", "Analysis", "Reports", "Recommendations", "About"])

    # common controls for Analysis page
    show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
    overlay_alpha = st.sidebar.slider("Heatmap intensity", 0.0, 1.0, 0.5)
    show_cluster = st.sidebar.checkbox("Show cluster plot", value=True)
    show_features = st.sidebar.checkbox("Show features chart", value=True)
    top_k = st.sidebar.slider("Show top-K features", 5, 100, 10)

    patients = load_patients()
    analyses = load_analyses()

    # ---------- Dashboard Page ----------
    if page == "Dashboard":
        st.markdown("<div class='section-title'><h2>Clinical Operations Dashboard</h2><p>Executive summary of patients, scans, and workflow performance.</p></div>", unsafe_allow_html=True)
        total_patients = len(patients)
        total_scans = len(analyses)
        avg_score = float(np.mean([a.get("score", 0) for a in analyses])) if analyses else 0.0
        deleted_patients_count = len(load_deleted_patients())

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-block'><h4>Total Patients</h4><span>{total_patients}</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-block'><h4>Total Scans</h4><span>{total_scans}</span></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-block'><h4>Avg Abnormality Score</h4><span>{avg_score:.3f}</span></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-block'><h4>Deleted Records</h4><span>{deleted_patients_count}</span></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Score distribution</h3><p>Review the current risk distribution across all scanned X-rays.</p>", unsafe_allow_html=True)
        if analyses:
            scores = [float(a.get("score", 0)) for a in analyses]
            hist_values, bins = np.histogram(scores, bins=10)
            st.bar_chart(hist_values)
        else:
            st.info("No analyses yet")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Recent Analyses</h3><p>Latest scan submissions and report generation activity.</p>", unsafe_allow_html=True)
        if analyses:
            df = pd.DataFrame(analyses)
            df["time_str"] = pd.to_datetime(df["time"], unit='s')
            st.dataframe(df[["time_str", "patient_id", "file", "score", "report"]].sort_values("time_str", ascending=False).head(20))
        else:
            st.write("No analyses available")
        st.markdown("</div>", unsafe_allow_html=True)

        deleted_patients = load_deleted_patients()
        show_deleted_history = st.checkbox("Show deleted patient history", value=False)
        if show_deleted_history:
            st.markdown("<div class='card'><h3>Deleted patient history</h3><p>Track removed patient profiles for audit and follow-up review.</p>", unsafe_allow_html=True)
            if deleted_patients:
                df_del = pd.DataFrame(deleted_patients)
                df_del["deleted_at"] = pd.to_datetime(df_del["deleted_at"], unit='s')
                st.dataframe(df_del[["deleted_at", "patient_id", "name", "age", "gender", "history"]].head(50))
            else:
                st.write("No deleted patient records yet")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Patient Registration Page ----------
    elif page == "Patients":
        st.markdown("<div class='section-title'><h2>Patient Registry</h2><p>Maintain structured patient profiles and clinical notes for each chest X-ray referral.</p></div>", unsafe_allow_html=True)
        with st.container():
            left, right = st.columns([2, 1])
            with left:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                patient_options = ["New patient"] + [f"{p.get('patient_id')} - {p.get('name')}" for p in patients]
                edit_selection = st.selectbox("Choose a patient to modify", patient_options)
                if edit_selection != "New patient":
                    selected_id = edit_selection.split(" - ")[0]
                    selected_patient = get_patient(patients, selected_id)
                    selected_index = find_patient_index(patients, selected_id)
                else:
                    selected_patient = None
                    selected_index = None

                with st.form("patient_form"):
                    pid = st.text_input("Patient ID", value=selected_patient.get("patient_id") if selected_patient else "")
                    name = st.text_input("Name", value=selected_patient.get("name") if selected_patient else "")
                    age = st.number_input("Age", min_value=0, max_value=120, value=selected_patient.get("age") if selected_patient else 30)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(selected_patient.get("gender")) if selected_patient else 0)
                    history_text = st.text_area("Medical History / Notes", value=selected_patient.get("history") if selected_patient else "")
                    save_clicked = st.form_submit_button("Save / Update Patient")

                if save_clicked:
                    if not pid or not name:
                        st.error("Patient ID and Name are required")
                    else:
                        patients = load_patients()
                        duplicate_exists = any(
                            p.get("patient_id") == pid and (not selected_patient or p.get("patient_id") != selected_patient.get("patient_id"))
                            for p in patients
                        )
                        if duplicate_exists:
                            st.error("This Patient ID is already registered. Please use a unique Patient ID.")
                        else:
                            record = {"patient_id": pid, "name": name, "age": int(age), "gender": gender, "history": history_text}
                            patients = [p for p in patients if p.get("patient_id") != pid]
                            patients.append(record)
                            save_patients(patients)
                            if selected_patient and selected_patient.get("patient_id") == pid:
                                st.success("Patient record updated successfully")
                            else:
                                st.success("New patient profile added")
                            patients = load_patients()

                if selected_patient:
                    if st.button("Delete Patient"):
                        patients = load_patients()
                        if selected_index is not None:
                            deleted_record = {
                                "deleted_at": int(time.time()),
                                "patient_id": selected_patient.get("patient_id"),
                                "name": selected_patient.get("name"),
                                "age": selected_patient.get("age"),
                                "gender": selected_patient.get("gender"),
                                "history": selected_patient.get("history"),
                            }
                            patients.pop(selected_index)
                            save_patients(patients)
                            append_deleted_patient(deleted_record)
                            st.success("Patient record deleted")
                            st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with right:
                st.markdown("<div class='card'><h3>Registry guidance</h3><ul><li>Select a patient entry to review or update clinical notes.</li><li>Keep Patient ID unique for audit and traceability.</li><li>Use Delete Patient to archive outdated profiles and retain history.</li></ul></div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Registered patient roster</h3>", unsafe_allow_html=True)
        if patients:
            st.dataframe(pd.DataFrame(patients))
        else:
            st.write("No patients registered yet")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Analysis Page ----------
    elif page == "Analysis":
        st.markdown("<div class='section-title'><h2>Scan Analysis</h2><p>Upload chest X-rays and review structured abnormality scoring with visual patient context.</p></div>", unsafe_allow_html=True)
        if not patients:
            st.warning("No patients found — please register a patient first.")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        selected_patient = st.selectbox("Select Patient", [p.get("patient_id") for p in patients] if patients else [])
        uploaded = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"]) 

        if uploaded and selected_patient:
            path = save_upload(uploaded)

            with st.spinner("Extracting features from model..."):
                feature_dim, chart_values, features = extract_features_cached(path)

            score = float(round(sum(features[:20]) / 20, 3))

            st.metric("Abnormality Score", value=score)
            if score >= 0:
                progress_score = min(max(score, 0.0), 1.0)
                st.progress(progress_score)
                st.caption("Score normalized to [0,1] where higher values indicate more abnormality signal.")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("<div class='metric-block'><h4>Original Image</h4></div>", unsafe_allow_html=True)
                img = load_image(path)
                if img is not None:
                    st.image(img, use_column_width=True)
            with col2:
                if show_heatmap:
                    st.markdown("<div class='metric-block'><h4>Heatmap Overlay</h4></div>", unsafe_allow_html=True)
                    heatmap_path = generate_heatmap(path)
                    blended = overlay_images(path, heatmap_path, alpha=overlay_alpha)
                    if blended is not None:
                        st.image(blended, use_column_width=True)

            if show_features:
                st.markdown("<div class='card'><h3>Top features</h3>", unsafe_allow_html=True)
                k = min(top_k, len(features))
                top_feats = [float(x) for x in features[:k]]
                df = pd.DataFrame({"feature_index": list(range(k)), "value": top_feats})
                st.dataframe(df)
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button("Download features CSV", data=csv_buf.getvalue(), file_name="features.csv", mime="text/csv")
                st.markdown("</div>", unsafe_allow_html=True)

            if show_cluster:
                st.markdown("<div class='card'><h3>Feature Clustering</h3>", unsafe_allow_html=True)
                cluster_path = generate_cluster_plot(features)
                cl = load_image(cluster_path)
                if cl is not None:
                    st.image(cl, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Create report and save analysis record
            report_id = f"RPT-{int(time.time())}"
            patient_record = get_patient(patients, selected_patient)
            report_recommendations = get_patient_recommendations(patient_record, score)

            with st.spinner("Preparing PDF report..."):
                report_name = create_report(
                    path,
                    feature_dim,
                    score=score,
                    patient_info=patient_record,
                    features=features,
                    recommendations=report_recommendations,
                    report_id=report_id,
                )
                report_path = os.path.join(REPORT_FOLDER, report_name)

            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button(label="Download PDF Report", data=f.read(), file_name=report_name, mime="application/pdf")

            # save analysis record
            analyses = load_analyses()
            record = {
                "time": int(time.time()),
                "patient_id": selected_patient,
                "file": os.path.basename(path),
                "score": score,
                "report": report_name,
                "report_id": report_id,
            }
            analyses.insert(0, record)
            save_analyses(analyses)
            st.success("Analysis saved")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Reports Page ----------
    elif page == "Reports":
        st.markdown("<div class='section-title'><h2>Report Center</h2><p>Access all generated scan reports and export patient documentation securely.</p></div>", unsafe_allow_html=True)
        analyses = load_analyses()
        if analyses:
            st.markdown("<div class='card'><h3>Report activity</h3><p>Recent PDF generation events are shown below for audit and review.</p></div>", unsafe_allow_html=True)
            for a in analyses:
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a.get('time', 0)))
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                cols = st.columns([3, 1])
                cols[0].markdown(
                    f"**{t}**  <br>Patient: **{a.get('patient_id')}**  <br>Report ID: **{a.get('report_id', 'N/A')}**  <br>Score: **{a.get('score')}**",
                    unsafe_allow_html=True,
                )
                rpt = a.get('report')
                rpt_path = os.path.join(REPORT_FOLDER, rpt)
                if os.path.exists(rpt_path):
                    with open(rpt_path, 'rb') as f:
                        cols[1].download_button(label="Download", data=f.read(), file_name=rpt, mime='application/pdf')
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No reports yet")

    # ---------- Recommendations Page ----------
    elif page == "Recommendations":
        st.markdown("<div class='section-title'><h2>Patient-specific Recommendations</h2><p>Generate guidance based on patient profile and the most recent scan.</p></div>", unsafe_allow_html=True)
        if not patients:
            st.info("No patients registered yet. Add patients in Patient Registration first.")
        else:
            patient_ids = [p.get("patient_id") for p in patients]
            selected_patient_id = st.selectbox("Select Patient", patient_ids)
            patient = get_patient(patients, selected_patient_id)
            analyses_for_patient = [a for a in analyses if a.get("patient_id") == selected_patient_id]
            last_score = None
            if analyses_for_patient:
                last_score = float(sorted(analyses_for_patient, key=lambda x: x.get("time", 0), reverse=True)[0].get("score", 0))

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader(f"Patient: {patient.get('name', selected_patient_id)}")
            st.write(f"**Age:** {patient.get('age', 'N/A')}  ")
            st.write(f"**Gender:** {patient.get('gender', 'N/A')}  ")
            st.write(f"**History:** {patient.get('history', 'N/A')}  ")
            if last_score is not None:
                st.write(f"**Latest score:** {last_score:.3f}")
            else:
                st.write("**Latest score:** No scan yet")

            st.markdown("<hr />", unsafe_allow_html=True)
            st.markdown("<h4>Recommended actions</h4>", unsafe_allow_html=True)
            recs = get_patient_recommendations(patient, last_score)
            for rec in recs:
                st.markdown(f"- {rec}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- About Page ----------
    else:
        st.markdown("<div class='section-title'><h2>About this platform</h2><p>Designed for modern diagnostic teams that require structured imaging workflows and actionable insights.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='about-grid'>", unsafe_allow_html=True)
        st.markdown("<div class='about-card'><div class='about-badge'>Operational focus</div><h3>Streamlined clinical workflow</h3><p>Supports patient intake, imaging analysis, report delivery, and ongoing follow-up in a single, cohesive interface.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='about-card'><div class='about-badge'>Clinical insight</div><h3>Data-driven prioritization</h3><p>Delivers abnormality scoring, heatmap visualization, and patient-specific recommendations for faster case triage.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='about-card'><div class='about-badge'>Professional delivery</div><h3>Audit-ready reporting</h3><p>Generates downloadable PDF reports for documentation and hand-off to referring providers or care teams.</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='about-card'><h3>Key advantages</h3><ul><li>Enables consistent patient registration and history tracking.</li><li>Pairs imaging scores with clinical recommendations for guided decision-making.</li><li>Offers secure export of reports and analysis outcomes for review.</li></ul></div>", unsafe_allow_html=True)
        st.markdown(r"""<div class='about-card'><svg class='animated-icon' viewBox='0 0 240 160' xmlns='http://www.w3.org/2000/svg'>
            <circle class='pulse-circle' cx='120' cy='80' r='52' fill='rgba(56,189,248,0.18)' />
            <path d='M70 95c6 2 14 4 24 4s18-2 24-4' fill='none' stroke='#2563eb' stroke-width='4' stroke-linecap='round' />
            <path d='M90 65c5-5 12-8 20-8s15 3 20 8' fill='none' stroke='#0f172a' stroke-width='4' stroke-linecap='round' />
            <rect x='110' y='48' width='36' height='44' rx='10' fill='#38bdf8' opacity='0.6'/>
            <circle cx='120' cy='80' r='10' fill='#0f172a' />
            <circle cx='130' cy='60' r='6' fill='#f8fafc' />
        </svg><p style='text-align:center; color:#334155; font-size:15px; margin:0;'>Built for clinical teams that demand reliability and clear imaging insight.</p></div>""", unsafe_allow_html=True)

    # ---------- About Page ----------


if __name__ == "__main__":
    main()
