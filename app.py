from flask import Flask, render_template, request, redirect, send_file, session, url_for, flash
import sqlite3
import os
import pickle
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
print("Importing config...", flush=True)
import config
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ---------------- DATABASE ----------------
print("Setting up DB...", flush=True)
os.makedirs("database", exist_ok=True)
conn = sqlite3.connect(config.DATABASE_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
user_id INTEGER,
description TEXT,
age INTEGER,
gender INTEGER,
dosage INTEGER,
risk TEXT
)
""")
try:
    cursor.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")
    conn.commit()
except:
    pass
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
age INTEGER,
email TEXT UNIQUE,
phone TEXT,
password TEXT
)
""")
conn.commit()

# ---------------- LOAD MODEL ----------------
print("Loading TF model...", flush=True)
model = load_model(config.MODEL_PATH)
print("Loaded TF model.", flush=True)

print("Loading tokenizer...", flush=True)
with open(config.TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Loading scaler...", flush=True)
with open(config.SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------------- LOAD ML MODELS ----------------
print("Loading ML models...", flush=True)

ml_models_path = "model/ml_models.pkl"
ml_metrics_path = "model/ml_metrics.json"

ml_models = {}
ml_metrics = {}

if os.path.exists(ml_models_path):
    with open(ml_models_path, 'rb') as f:
        ml_models = pickle.load(f)

if os.path.exists(ml_metrics_path):
    with open(ml_metrics_path, 'r') as f:
        ml_metrics = json.load(f)

side_effects = [
    "Nausea","Headache","Dizziness","Fatigue","Rash",
    "Vomiting","Insomnia","Anxiety","Fever","Dry Mouth"
]

# ---------------- DRUG ALTERNATIVES DATABASE ----------------
# Maps lowercase keywords to a list of candidate safer alternatives
DRUG_ALTERNATIVES_DB = {
    # NSAIDs / Pain relievers
    'aspirin':      ['Paracetamol', 'Acetaminophen', 'Naproxen'],
    'ibuprofen':    ['Paracetamol', 'Celecoxib', 'Naproxen'],
    'diclofenac':   ['Paracetamol', 'Ibuprofen', 'Celecoxib'],
    'naproxen':     ['Paracetamol', 'Ibuprofen', 'Celecoxib'],
    'ketoprofen':   ['Paracetamol', 'Ibuprofen', 'Naproxen'],
    'ketorolac':    ['Paracetamol', 'Ibuprofen', 'Celecoxib'],
    # Paracetamol / Fever
    'paracetamol':  ['Ibuprofen', 'Aspirin', 'Naproxen'],
    'acetaminophen':['Ibuprofen', 'Aspirin', 'Celecoxib'],
    'dolo':         ['Crocin', 'Paracetamol', 'Ibuprofen'],
    'crocin':       ['Dolo', 'Paracetamol', 'Ibuprofen'],
    # Antibiotics
    'amoxicillin':  ['Azithromycin', 'Doxycycline', 'Cephalexin'],
    'azithromycin': ['Amoxicillin', 'Doxycycline', 'Clindamycin'],
    'ciprofloxacin':['Doxycycline', 'Amoxicillin', 'Cephalexin'],
    'erythromycin': ['Azithromycin', 'Clarithromycin', 'Doxycycline'],
    'doxycycline':  ['Azithromycin', 'Amoxicillin', 'Minocycline'],
    # Antacids / GI
    'omeprazole':   ['Pantoprazole', 'Ranitidine', 'Esomeprazole'],
    'pantoprazole': ['Omeprazole', 'Ranitidine', 'Esomeprazole'],
    'ranitidine':   ['Omeprazole', 'Pantoprazole', 'Famotidine'],
    # Antihistamines
    'cetirizine':   ['Loratadine', 'Fexofenadine', 'Levocetirizine'],
    'loratadine':   ['Cetirizine', 'Fexofenadine', 'Bilastine'],
    'diphenhydramine':['Cetirizine', 'Loratadine', 'Fexofenadine'],
    # Statins
    'atorvastatin': ['Rosuvastatin', 'Pravastatin', 'Simvastatin'],
    'simvastatin':  ['Atorvastatin', 'Rosuvastatin', 'Pravastatin'],
    # Antidiabetics
    'metformin':    ['Glipizide', 'Sitagliptin', 'Glimepiride'],
    'glipizide':    ['Metformin', 'Sitagliptin', 'Glimepiride'],
    # Antihypertensives
    'amlodipine':   ['Lisinopril', 'Losartan', 'Atenolol'],
    'lisinopril':   ['Amlodipine', 'Losartan', 'Enalapril'],
    'atenolol':     ['Amlodipine', 'Metoprolol', 'Bisoprolol'],
}
# General fallback candidates when drug is unknown
FALLBACK_ALTERNATIVES = ['Paracetamol', 'Ibuprofen', 'Cetirizine', 'Omeprazole', 'Azithromycin']

# ---------------- SIDE EFFECT TIMELINE DATABASE ----------------
# Clinically estimated onset windows and duration for each predicted side effect.
# onset_min/max: hours after drug ingestion before effect may appear.
# dur_min/max:   hours the effect may last (from onset_max).
# Sources: clinical pharmacology textbooks, package inserts, WHO ADR database.
SIDE_EFFECT_TIMELINE_DB = {
    'Nausea':    {'onset_min': 0.5, 'onset_max': 4,   'dur_min': 2,  'dur_max': 8,
                  'onset_label': '30 min – 4 hrs',  'duration_label': '2 – 8 hrs',
                  'tip': 'Take with food to reduce nausea. Ginger tea may help.'},
    'Headache':  {'onset_min': 1,   'onset_max': 6,   'dur_min': 2,  'dur_max': 12,
                  'onset_label': '1 – 6 hrs',        'duration_label': '2 – 12 hrs',
                  'tip': 'Stay hydrated. Rest in a quiet, dark room if severe.'},
    'Dry Mouth': {'onset_min': 0.5, 'onset_max': 2,   'dur_min': 2,  'dur_max': 8,
                  'onset_label': '30 min – 2 hrs',   'duration_label': '2 – 8 hrs',
                  'tip': 'Sip water frequently. Avoid caffeine and alcohol.'},
    'Fatigue':   {'onset_min': 1,   'onset_max': 6,   'dur_min': 4,  'dur_max': 24,
                  'onset_label': '1 – 6 hrs',        'duration_label': '4 – 24 hrs',
                  'tip': 'Avoid driving or operating machinery if fatigue is severe.'},
    'Dizziness': {'onset_min': 0.5, 'onset_max': 2,   'dur_min': 1,  'dur_max': 8,
                  'onset_label': '30 min – 2 hrs',   'duration_label': '1 – 8 hrs',
                  'tip': 'Rise slowly from sitting or lying. Avoid sudden movements.'},
    'Insomnia':  {'onset_min': 4,   'onset_max': 8,   'dur_min': 8,  'dur_max': 24,
                  'onset_label': '4 – 8 hrs',        'duration_label': '8 – 24 hrs',
                  'tip': 'Take the drug in the morning if possible. Avoid caffeine after noon.'},
    'Anxiety':   {'onset_min': 1,   'onset_max': 4,   'dur_min': 2,  'dur_max': 12,
                  'onset_label': '1 – 4 hrs',        'duration_label': '2 – 12 hrs',
                  'tip': 'Practice deep breathing. Discuss dose adjustment with your doctor.'},
    'Rash':      {'onset_min': 24,  'onset_max': 72,  'dur_min': 24, 'dur_max': 120,
                  'onset_label': '24 – 72 hrs',      'duration_label': '1 – 5 days',
                  'tip': 'Stop medication immediately if rash is spreading or blistering. Seek medical advice.'},
    'Vomiting':  {'onset_min': 0.5, 'onset_max': 3,   'dur_min': 1,  'dur_max': 8,
                  'onset_label': '30 min – 3 hrs',   'duration_label': '1 – 8 hrs',
                  'tip': 'Stay hydrated. Small sips of clear fluid. Seek care if persistent.'},
    'Fever':     {'onset_min': 2,   'onset_max': 24,  'dur_min': 12, 'dur_max': 72,
                  'onset_label': '2 – 24 hrs',       'duration_label': '12 hrs – 3 days',
                  'tip': 'Drug-induced fever is usually benign; stop drug if temperature exceeds 39.5 °C.'},
}

def build_timeline_data(model_results: dict, prob_threshold: float = 0.25) -> list:
    """
    Enrich model predictions with timeline data.
    Only include effects with adjusted probability >= prob_threshold.
    Returns list sorted by onset_min (earliest appearing effects first).
    """
    timeline = []
    for effect_name, details in model_results.items():
        if details['probability'] < prob_threshold:
            continue
        tl = SIDE_EFFECT_TIMELINE_DB.get(effect_name)
        if not tl:
            continue
        # Timeline absolute end = onset_max + dur_max hours after ingestion
        abs_end = tl['onset_max'] + tl['dur_max']
        timeline.append({
            'name':           effect_name,
            'severity':       details['severity'],
            'probability':    round(details['probability'] * 100, 1),
            'onset_min':      tl['onset_min'],
            'onset_max':      tl['onset_max'],
            'dur_min':        tl['dur_min'],
            'dur_max':        tl['dur_max'],
            'onset_label':    tl['onset_label'],
            'duration_label': tl['duration_label'],
            'abs_end':        abs_end,
            'tip':            tl['tip'],
        })
    timeline.sort(key=lambda x: x['onset_min'])
    return timeline


# ---------------- CRITICAL ALERT ENGINE ----------------
# Thresholds (probability 0.0–1.0) for escalating alert tiers.
# CRITICAL  ≥ 40%   — Life-threatening; immediate medical attention required
# WARNING   ≥ 25%   — Serious risk; consult doctor before use
# ELEVATED  ≥ 15%   — Notable life-threatening signal; monitor closely
CRITICAL_THRESHOLD  = 0.40
WARNING_THRESHOLD   = 0.25
ELEVATED_THRESHOLD  = 0.15

def detect_critical_alerts(model_results: dict) -> list:
    """
    Scan model results for Life-threatening effects that breach any alert tier.
    Returns list of alert dicts sorted by probability descending (highest risk first).
    Each dict contains: name, probability (%), level, icon, guidance, bg/text/border colours.
    """
    alerts = []
    for effect_name, details in model_results.items():
        if details['severity'] != 'Life-threatening':
            continue
        prob = details['probability']
        if prob < ELEVATED_THRESHOLD:
            continue

        if prob >= CRITICAL_THRESHOLD:
            level    = 'CRITICAL'
            icon     = '🚨'
            guidance = (
                f'<strong>IMMEDIATE ACTION REQUIRED.</strong> '
                f'The AI model has detected a <strong>{round(prob*100,1)}%</strong> probability of '
                f'<em>{effect_name}</em> — a life-threatening reaction. '
                f'Do NOT take this medication without first consulting a doctor or pharmacist. '
                f'If you have already taken the drug and feel unwell, call emergency services immediately.'
            )
            bg, bd, txt = 'rgba(239,68,68,0.12)', 'rgba(239,68,68,0.55)', '#fca5a5'
        elif prob >= WARNING_THRESHOLD:
            level    = 'WARNING'
            icon     = '⚠️'
            guidance = (
                f'<strong>MEDICAL CONSULTATION REQUIRED.</strong> '
                f'The model predicts a <strong>{round(prob*100,1)}%</strong> chance of '
                f'<em>{effect_name}</em>, which is classified as life-threatening. '
                f'Please discuss this risk with your doctor before taking this medication. '
                f'Do not ignore any unusual symptoms.'
            )
            bg, bd, txt = 'rgba(249,115,22,0.10)', 'rgba(249,115,22,0.50)', '#fdba74'
        else:
            level    = 'ELEVATED'
            icon     = '🔶'
            guidance = (
                f'<strong>ELEVATED RISK DETECTED.</strong> '
                f'A <strong>{round(prob*100,1)}%</strong> probability of '
                f'<em>{effect_name}</em> has been identified. '
                f'While not an emergency, this warrants a conversation with your healthcare provider. '
                f'Monitor for any early warning symptoms after taking the medication.'
            )
            bg, bd, txt = 'rgba(234,179,8,0.10)', 'rgba(234,179,8,0.45)', '#fde047'

        alerts.append({
            'name':        effect_name,
            'probability': round(prob * 100, 1),
            'level':       level,
            'icon':        icon,
            'guidance':    guidance,
            'bg':          bg,
            'bd':          bd,
            'txt':         txt,
        })

    alerts.sort(key=lambda x: -x['probability'])
    return alerts


#          WHO Model Formulary, ClinicalPharmacology database.
# Structure: (drug_a_lowercase, drug_b_lowercase) -> interaction dict
# Severity: CONTRAINDICATED > SEVERE > MODERATE > MILD
DDI_DATABASE = [
    # ── Aspirin interactions ──────────────────────────────────────────────────
    {
        'drugs': {'aspirin', 'ibuprofen'},
        'severity': 'SEVERE',
        'effect': 'Increased risk of gastrointestinal bleeding and reduced antiplatelet effect of aspirin.',
        'mechanism': 'Ibuprofen competitively blocks COX-1, preventing aspirin from binding and reducing its antiplatelet benefit.',
        'recommendation': 'Avoid combination. If NSAID needed, use paracetamol or a COX-2 inhibitor (celecoxib) with aspirin.',
        'alternatives': ['Paracetamol', 'Celecoxib']
    },
    {
        'drugs': {'aspirin', 'warfarin'},
        'severity': 'CONTRAINDICATED',
        'effect': 'Greatly increased bleeding risk. Can cause life-threatening haemorrhage.',
        'mechanism': 'Aspirin inhibits platelet aggregation; warfarin inhibits clotting factors. Combined effect is synergistic.',
        'recommendation': 'Do NOT combine unless under close specialist supervision (e.g., mechanical heart valve). Use lowest aspirin dose possible.',
        'alternatives': ['Paracetamol']
    },
    {
        'drugs': {'aspirin', 'methotrexate'},
        'severity': 'SEVERE',
        'effect': 'Aspirin reduces kidney clearance of methotrexate, causing toxic methotrexate levels.',
        'mechanism': 'Competition for renal tubular secretion reduces methotrexate excretion.',
        'recommendation': 'Avoid combination. Monitor methotrexate levels closely if unavoidable.',
        'alternatives': ['Paracetamol']
    },
    # ── Ibuprofen interactions ────────────────────────────────────────────────
    {
        'drugs': {'ibuprofen', 'lisinopril'},
        'severity': 'MODERATE',
        'effect': 'NSAIDs reduce the blood-pressure lowering effect of ACE inhibitors and may worsen kidney function.',
        'mechanism': 'NSAIDs cause sodium/water retention and reduce renal prostaglandins, opposing ACE inhibitor action.',
        'recommendation': 'Use paracetamol instead. Monitor blood pressure and renal function if combination is necessary.',
        'alternatives': ['Paracetamol', 'Celecoxib']
    },
    {
        'drugs': {'ibuprofen', 'warfarin'},
        'severity': 'SEVERE',
        'effect': 'Significantly increased risk of bleeding, including GI haemorrhage.',
        'mechanism': 'Ibuprofen displaces warfarin from protein binding and irritates GI mucosa.',
        'recommendation': 'Substitute ibuprofen with paracetamol. Closely monitor INR if combination cannot be avoided.',
        'alternatives': ['Paracetamol']
    },
    {
        'drugs': {'ibuprofen', 'metformin'},
        'severity': 'MODERATE',
        'effect': 'Increased risk of metformin-induced lactic acidosis due to reduced kidney blood flow.',
        'mechanism': 'NSAIDs reduce renal prostaglandins, decreasing GFR and delaying metformin excretion.',
        'recommendation': 'Use paracetamol if pain relief is needed in patients on metformin.',
        'alternatives': ['Paracetamol']
    },
    {
        'drugs': {'ibuprofen', 'atenolol'},
        'severity': 'MODERATE',
        'effect': 'Reduced antihypertensive effect of beta-blockers.',
        'mechanism': 'NSAIDs promote sodium retention, opposing beta-blocker-induced reduction in blood pressure.',
        'recommendation': 'Monitor blood pressure. Prefer paracetamol for pain/fever.',
        'alternatives': ['Paracetamol']
    },
    # ── Antibiotic interactions ───────────────────────────────────────────────
    {
        'drugs': {'amoxicillin', 'methotrexate'},
        'severity': 'SEVERE',
        'effect': 'Amoxicillin can raise methotrexate levels to toxic concentrations.',
        'mechanism': 'Penicillins compete for renal tubular secretion of methotrexate.',
        'recommendation': 'Monitor methotrexate levels closely. Consider alternative antibiotic (azithromycin).',
        'alternatives': ['Azithromycin', 'Doxycycline']
    },
    {
        'drugs': {'amoxicillin', 'warfarin'},
        'severity': 'MODERATE',
        'effect': 'Amoxicillin may increase anticoagulant effect of warfarin, raising bleeding risk.',
        'mechanism': 'Reduced intestinal bacteria decreases vitamin K synthesis, potentiating warfarin.',
        'recommendation': 'Monitor INR more frequently during antibiotic course and for 1 week after.',
        'alternatives': ['Azithromycin — same caution applies; monitor INR regardless']
    },
    {
        'drugs': {'azithromycin', 'atorvastatin'},
        'severity': 'MODERATE',
        'effect': 'Increased risk of muscle injury (myopathy/rhabdomyolysis).',
        'mechanism': 'Azithromycin inhibits CYP3A4, raising atorvastatin plasma levels.',
        'recommendation': 'Use a short course only. Temporarily holding statin during 5-day azithromycin course is prudent.',
        'alternatives': ['Amoxicillin', 'Doxycycline']
    },
    {
        'drugs': {'azithromycin', 'ciprofloxacin'},
        'severity': 'SEVERE',
        'effect': 'Additive QT prolongation — risk of fatal ventricular arrhythmia (Torsades de Pointes).',
        'mechanism': 'Both drugs independently prolong the cardiac QT interval via hERG potassium channel blockade.',
        'recommendation': 'Do NOT combine. Use only one QT-prolonging antibiotic at a time.',
        'alternatives': ['Amoxicillin', 'Doxycycline']
    },
    {
        'drugs': {'ciprofloxacin', 'antacid'},
        'severity': 'MODERATE',
        'effect': 'Antacids reduce absorption of ciprofloxacin by up to 90%.',
        'mechanism': 'Divalent/trivalent cations (Mg²⁺, Al³⁺, Ca²⁺) chelate ciprofloxacin, forming insoluble complexes.',
        'recommendation': 'Take ciprofloxacin 2 hours before or 6 hours after antacids.',
        'alternatives': []
    },
    {
        'drugs': {'doxycycline', 'antacid'},
        'severity': 'MODERATE',
        'effect': 'Metal ions in antacids form chelates with doxycycline, dramatically reducing absorption.',
        'mechanism': 'Chelation by Ca²⁺, Mg²⁺, Fe²⁺ ions prevents doxycycline absorption.',
        'recommendation': 'Take doxycycline 2 hours before or after antacids, dairy, or iron supplements.',
        'alternatives': []
    },
    # ── Antihistamine interactions ────────────────────────────────────────────
    {
        'drugs': {'cetirizine', 'loratadine'},
        'severity': 'MILD',
        'effect': 'Additive antihistamine effect with potential for increased sedation.',
        'mechanism': 'Both compete for H1 receptors — combining provides no additional benefit and increases side effects.',
        'recommendation': 'Use only one antihistamine at a time. Choose loratadine (less sedating) or cetirizine.',
        'alternatives': ['Loratadine', 'Fexofenadine']
    },
    {
        'drugs': {'diphenhydramine', 'cetirizine'},
        'severity': 'MODERATE',
        'effect': 'Excessive sedation, cognitive impairment, and increased fall risk.',
        'mechanism': 'Both are H1 antagonists; diphenhydramine also has central anticholinergic properties.',
        'recommendation': 'Do not combine sedating and non-sedating antihistamines. Use one at a time.',
        'alternatives': ['Loratadine', 'Fexofenadine']
    },
    # ── Antacid / PPI interactions ────────────────────────────────────────────
    {
        'drugs': {'omeprazole', 'clopidogrel'},
        'severity': 'SEVERE',
        'effect': 'Omeprazole significantly reduces the antiplatelet effect of clopidogrel.',
        'mechanism': 'Omeprazole inhibits CYP2C19, the enzyme that converts clopidogrel to its active form.',
        'recommendation': 'Use pantoprazole instead of omeprazole (minimal CYP2C19 inhibition) with clopidogrel.',
        'alternatives': ['Pantoprazole', 'Ranitidine']
    },
    # ── Statin interactions ───────────────────────────────────────────────────
    {
        'drugs': {'simvastatin', 'amoxicillin'},
        'severity': 'MILD',
        'effect': 'Minimal interaction; slight theoretical risk of altered statin levels.',
        'mechanism': 'Low clinical significance in practice.',
        'recommendation': 'No dose adjustment needed. Standard monitoring applies.',
        'alternatives': []
    },
    {
        'drugs': {'atorvastatin', 'amlodipine'},
        'severity': 'MILD',
        'effect': 'Amlodipine mildly increases atorvastatin exposure (~18%).',
        'mechanism': 'Amlodipine mildly inhibits CYP3A4, slightly raising atorvastatin levels.',
        'recommendation': 'Safe combination; keep atorvastatin dose ≤40 mg/day. Monitor for muscle symptoms.',
        'alternatives': []
    },
    # ── Antidiabetic interactions ─────────────────────────────────────────────
    {
        'drugs': {'metformin', 'lisinopril'},
        'severity': 'MILD',
        'effect': 'Lisinopril may cause a modest improvement in insulin sensitivity — usually beneficial.',
        'mechanism': 'ACE inhibition reduces angiotensin II, improving glucose metabolism.',
        'recommendation': 'Beneficial combination in diabetic hypertension. Monitor for hypoglycaemia.',
        'alternatives': []
    },
    {
        'drugs': {'metformin', 'ciprofloxacin'},
        'severity': 'MODERATE',
        'effect': 'Ciprofloxacin can cause both hypoglycaemia and hyperglycaemia in diabetic patients on metformin.',
        'mechanism': 'Fluoroquinolones stimulate insulin secretion and may potentiate glucose-lowering.',
        'recommendation': 'Monitor blood glucose closely during ciprofloxacin therapy.',
        'alternatives': ['Amoxicillin', 'Azithromycin']
    },
    # ── Antihypertensive combinations ────────────────────────────────────────
    {
        'drugs': {'lisinopril', 'losartan'},
        'severity': 'SEVERE',
        'effect': 'Dual renin-angiotensin blockade increases risk of hypotension, hyperkalaemia, and acute kidney injury.',
        'mechanism': 'Combining ACE inhibitor + ARB produces additive renin-angiotensin blockade without additional cardiovascular benefit.',
        'recommendation': 'Do NOT combine ACE inhibitors and ARBs routinely. Use one class only.',
        'alternatives': ['Amlodipine', 'Atenolol']
    },
    {
        'drugs': {'amlodipine', 'lisinopril'},
        'severity': 'MILD',
        'effect': 'Additive blood pressure lowering — generally beneficial in hypertension.',
        'mechanism': 'Calcium channel blocker + ACE inhibitor have complementary mechanisms.',
        'recommendation': 'Standard combination for hypertension. Monitor blood pressure and watch for hypotension.',
        'alternatives': []
    },
    # ── Paracetamol combinations ──────────────────────────────────────────────
    {
        'drugs': {'paracetamol', 'dolo'},
        'severity': 'CONTRAINDICATED',
        'effect': 'Double dosing of paracetamol — high risk of acute liver failure.',
        'mechanism': 'Dolo 650 contains paracetamol 650 mg. Combining with additional paracetamol exceeds safe daily limit (4000 mg).',
        'recommendation': 'Never take two paracetamol-containing products simultaneously.',
        'alternatives': ['Ibuprofen (if no contraindication)', 'Aspirin']
    },
    {
        'drugs': {'paracetamol', 'crocin'},
        'severity': 'CONTRAINDICATED',
        'effect': 'Double dosing of paracetamol — risk of hepatotoxicity and liver failure.',
        'mechanism': 'Crocin contains paracetamol. Combining exceeds maximum safe daily dose.',
        'recommendation': 'Use only one paracetamol-based product at a time.',
        'alternatives': ['Ibuprofen (if no contraindication)', 'Naproxen']
    },
    {
        'drugs': {'dolo', 'crocin'},
        'severity': 'CONTRAINDICATED',
        'effect': 'Double dosing of paracetamol — high risk of acute liver failure.',
        'mechanism': 'Both are paracetamol-based brands. Combined dose can easily exceed 4000 mg/day.',
        'recommendation': 'Never combine two paracetamol products. Choose one and respect the recommended dose.',
        'alternatives': ['Ibuprofen (if no contraindication)']
    },
    # ── Cetirizine (Zyrtec) combinations ────────────────────────────────────
    {
        'drugs': {'cetirizine', 'dolo'},
        'severity': 'MILD',
        'effect': 'No clinically significant pharmacokinetic interaction. Additive symptom relief is the goal.',
        'mechanism': 'Different mechanisms (H1 antagonist vs paracetamol COX inhibitor).',
        'recommendation': 'Safe to combine for cold/flu symptoms at standard doses.',
        'alternatives': []
    },
    {
        'drugs': {'cetirizine', 'amoxicillin'},
        'severity': 'MILD',
        'effect': 'No known significant interaction. Often deliberately combined for allergic infections.',
        'mechanism': 'Different mechanisms of action; no pharmacokinetic interaction documented.',
        'recommendation': 'Safe combination. Standard monitoring for individual drug side effects.',
        'alternatives': []
    },
]

def check_interactions(drug_list: list) -> list:
    """
    Given a list of drug names, return all pairwise DDI alerts found in DDI_DATABASE.
    Returns list of interaction dicts, sorted by severity (CONTRAINDICATED first).
    """
    normalised = [d.strip().lower() for d in drug_list if d.strip()]
    found = []
    severity_order = {'CONTRAINDICATED': 0, 'SEVERE': 1, 'MODERATE': 2, 'MILD': 3}

    for i_idx, drug_a in enumerate(normalised):
        for drug_b in normalised[i_idx + 1:]:
            if drug_a == drug_b:
                continue
            for entry in DDI_DATABASE:
                if {drug_a, drug_b} == entry['drugs']:
                    # Find display-capitalised names from original input
                    def cap(d):
                        for orig in drug_list:
                            if orig.strip().lower() == d:
                                return orig.strip().title()
                        return d.title()
                    found.append({
                        'drug_a':         cap(drug_a),
                        'drug_b':         cap(drug_b),
                        'severity':       entry['severity'],
                        'effect':         entry['effect'],
                        'mechanism':      entry['mechanism'],
                        'recommendation': entry['recommendation'],
                        'alternatives':   entry.get('alternatives', []),
                    })
                    break  # Only one entry per pair

    found.sort(key=lambda x: severity_order.get(x['severity'], 9))
    return found

# ---------------- DOSAGE SAFETY THRESHOLDS ----------------
# Max recommended single-dose (mg) per drug based on standard medical guidelines.
# Source: standard pharmacological references / WHO / drug package inserts.
# If dosage exceeds the threshold a risk multiplier is applied.
DRUG_DOSAGE_LIMITS = {
    # Pain / Fever
    'paracetamol': 1000, 'acetaminophen': 1000,
    'dolo': 1000, 'crocin': 1000,
    'ibuprofen': 400, 'diclofenac': 50,
    'naproxen': 500, 'aspirin': 650,
    'ketoprofen': 100, 'ketorolac': 30,
    # Antibiotics
    'amoxicillin': 500, 'azithromycin': 500,
    'ciprofloxacin': 500, 'erythromycin': 500,
    'doxycycline': 200, 'cephalexin': 500,
    'clindamycin': 450, 'minocycline': 100,
    # Antacids / GI
    'omeprazole': 40, 'pantoprazole': 40,
    'ranitidine': 150, 'esomeprazole': 40,
    'famotidine': 40,
    # Antihistamines
    'cetirizine': 10, 'loratadine': 10,
    'fexofenadine': 180, 'levocetirizine': 5,
    'diphenhydramine': 50, 'bilastine': 20,
    # Statins
    'atorvastatin': 80, 'rosuvastatin': 40,
    'simvastatin': 40, 'pravastatin': 80,
    # Antidiabetics
    'metformin': 1000, 'glipizide': 20,
    'sitagliptin': 100, 'glimepiride': 4,
    # Antihypertensives
    'amlodipine': 10, 'lisinopril': 40,
    'losartan': 100, 'atenolol': 100,
    'metoprolol': 200, 'bisoprolol': 10,
    'enalapril': 40,
}
# Generic fallback safe dose (mg) for unknown drugs
DEFAULT_SAFE_DOSE = 500

def get_dosage_multiplier(drug_name: str, dosage: int) -> float:
    """
    Returns a risk multiplier (>= 1.0) based on how far the dosage exceeds
    the drug's standard safe single-dose limit.

    Multiplier formula (linear scale):
      ratio = dosage / safe_limit
      ratio <= 1.0  → multiplier = 1.0   (within safe range, no boost)
      ratio 1–2     → multiplier = 1.0 to 1.4  (moderate overdose)
      ratio 2–4     → multiplier = 1.4 to 1.9  (significant overdose)
      ratio >  4    → multiplier = 2.0+         (dangerous overdose)
    """
    key = drug_name.strip().lower()
    safe_limit = DRUG_DOSAGE_LIMITS.get(key, DEFAULT_SAFE_DOSE)
    ratio = dosage / safe_limit
    if ratio <= 1.0:
        return 1.0
    elif ratio <= 2.0:
        # Linear from 1.0 → 1.4
        return 1.0 + 0.4 * (ratio - 1.0)
    elif ratio <= 4.0:
        # Linear from 1.4 → 1.9
        return 1.4 + 0.25 * (ratio - 2.0)
    else:
        # Cap multiplier at 2.0 for extreme overdoses
        return min(2.0, 1.9 + 0.05 * (ratio - 4.0))

def find_safer_alternatives(drug_name, age, gender, dosage, current_avg_risk):
    """Run model on candidate alternatives and return those safer than the current drug."""
    drug_key = drug_name.strip().lower()
    candidates = DRUG_ALTERNATIVES_DB.get(drug_key, FALLBACK_ALTERNATIVES)
    # Remove the current drug from candidates (case-insensitive)
    candidates = [c for c in candidates if c.lower() != drug_key]

    safer = []
    for alt in candidates:
        try:
            _, _, _, alt_avg_prob = predict(alt, age, gender, dosage)
            alt_risk_score = round(alt_avg_prob * 100, 1)
            current_risk_score = round(current_avg_risk * 100, 1)
            if alt_avg_prob < current_avg_risk:
                if alt_avg_prob < 0.3:      risk_badge = 'LOW'
                elif alt_avg_prob < 0.5:    risk_badge = 'MEDIUM'
                else:                       risk_badge = 'HIGH'
                safer.append({
                    'name': alt,
                    'risk_score': alt_risk_score,
                    'risk_badge': risk_badge,
                    'saved': round(current_risk_score - alt_risk_score, 1)
                })
        except Exception:
            continue

    # Sort by lowest risk first, return top 3
    safer.sort(key=lambda x: x['risk_score'])
    return safer[:3]

def predict(description, age, gender, dosage):
    seq = tokenizer.texts_to_sequences([description])
    text_pad = pad_sequences(seq, maxlen=20)

    structured = scaler.transform([[age, gender, dosage]])

    prediction = model.predict([text_pad, structured])[0]

    severity_mapping = {
        "Nausea": "Mild",
        "Headache": "Mild",
        "Dry Mouth": "Mild",
        "Fatigue": "Mild",
        "Dizziness": "Moderate",
        "Insomnia": "Moderate",
        "Anxiety": "Moderate",
        "Rash": "Severe",
        "Vomiting": "Severe",
        "Fever": "Life-threatening"
    }

    # ── Dosage risk modifier ──────────────────────────────────────────────
    # Apply a medically-grounded multiplier when dosage exceeds the drug's
    # standard safe single-dose limit. This ensures that inputting a
    # dangerously high dose correctly elevates all risk outputs.
    dose_mult = get_dosage_multiplier(description, dosage)

    result = {}
    for i, effect in enumerate(side_effects):
        raw_prob = float(prediction[i])
        # Scale probability by dosage multiplier, cap at 0.99
        adjusted_prob = min(0.99, raw_prob * dose_mult)
        result[effect] = {
            "severity": severity_mapping.get(effect, "Moderate"),
            "probability": adjusted_prob
        }

    # Use adjusted average for risk level classification
    adjusted_probs = [v["probability"] for v in result.values()]
    avg_risk = float(np.mean(adjusted_probs))

    if avg_risk > 0.7:
        risk_level = "HIGH"
    elif avg_risk > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # ML Probabilities — also scaled by dosage multiplier
    ml_probs = {}
    input_df = pd.DataFrame([{
        'description': description,
        'age': age,
        'gender': gender,
        'dosage': dosage
    }])

    for name, pipeline in ml_models.items():
        raw_ml_prob = pipeline.predict_proba(input_df)[0][1]
        # Apply dosage multiplier to ML probability as well, cap at 0.99
        ml_probs[name] = float(min(0.99, raw_ml_prob * dose_mult))

    avg_ml_prob = float(np.mean(list(ml_probs.values()))) if ml_probs else 0.0

    return result, risk_level, ml_probs, avg_ml_prob

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        description = request.form["description"]
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        dosage = int(request.form["dosage"])

        results, risk, ml_probs, avg_ml_prob = predict(description, age, gender, dosage)

        cursor.execute("INSERT INTO history(user_id, description,age,gender,dosage,risk) VALUES (?,?,?,?,?,?)",
                       (session['user_id'], description, age, gender, dosage, risk))
        conn.commit()

        # Find safer alternatives when risk is significant (avg_ml_prob > 0.50)
        safer_alternatives = []
        if avg_ml_prob > 0.50:
            safer_alternatives = find_safer_alternatives(description, age, gender, dosage, avg_ml_prob)

        # Generate real age-based risk comparisons across age groups (Child, Teen, Adult, Elderly)
        age_comparisons = []
        for test_age in [10, 15, 35, 70]:
            try:
                _, _, _, a_avg_ml = predict(description, test_age, gender, dosage)
                # Fallback to model's base avg_risk if ml_model has issues, but we'll use avg_ml_prob logic
                age_comparisons.append(round(a_avg_ml * 100, 1))
            except Exception:
                age_comparisons.append(0)

        # Build side-effect timeline enriched data
        timeline_data = build_timeline_data(results)

        # Detect critical / life-threatening alerts above threshold
        critical_alerts = detect_critical_alerts(results)

        return render_template("dashboard.html",
                               results=results,
                               risk=risk,
                               drug_name=description,
                               ml_probs=ml_probs,
                               avg_ml_prob=avg_ml_prob,
                               ml_metrics=ml_metrics,
                               safer_alternatives=safer_alternatives,
                               age_comparisons=age_comparisons,
                               timeline_data=timeline_data,
                               critical_alerts=critical_alerts)

    return render_template("dashboard.html", ml_metrics=ml_metrics)

@app.route("/ddi", methods=["GET", "POST"])
def ddi():
    if "user_id" not in session:
        return redirect(url_for("login"))
    interactions = []
    drug_inputs  = []
    checked      = False
    if request.method == "POST":
        # Collect up to 5 drug inputs
        drug_inputs = [request.form.get(f"drug_{i}", "").strip() for i in range(1, 6)]
        drug_inputs = [d for d in drug_inputs if d]   # Remove empty
        if len(drug_inputs) >= 2:
            interactions = check_interactions(drug_inputs)
        checked = True
    return render_template("ddi.html",
                           drug_inputs=drug_inputs,
                           interactions=interactions,
                           checked=checked)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]

        hashed = generate_password_hash(password)

        try:
            cursor.execute("INSERT INTO users(name, age, email, phone, password) VALUES (?, ?, ?, ?, ?)",
                           (name, age, email, phone, hashed))
            conn.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already exists!")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[5], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials!")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    cursor.execute("SELECT id, description, age, gender, dosage, risk FROM history WHERE user_id=?", (session['user_id'],))
    data = cursor.fetchall()
    return render_template("history.html", data=data)

@app.route("/export")
def export():
    if "user_id" not in session:
        return redirect(url_for("login"))
    cursor.execute("SELECT description,age,gender,dosage,risk FROM history WHERE user_id=?", (session['user_id'],))
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=["Description","Age","Gender","Dosage","Risk"])
    os.makedirs("reports", exist_ok=True)
    import time
    filename = f"history_{session['user_id']}_{int(time.time())}.xlsx"
    path = os.path.abspath(f"reports/{filename}")
    df.to_excel(path, index=False, engine="openpyxl")

    # In older Flask, attachment_filename is used. In newer, download_name. We use as_attachment and let it take the filename.
    return send_file(path, as_attachment=True)

@app.route("/report/<int:id>")
def report(id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    cursor.execute("SELECT id, description, age, gender, dosage, risk FROM history WHERE id=? AND user_id=?", (id, session['user_id']))
    row = cursor.fetchone()
    
    if not row:
        return "Record not found", 404

    os.makedirs("reports", exist_ok=True)
    import time
    filename = f"report_{id}_{int(time.time())}.pdf"
    path = os.path.abspath(f"reports/{filename}")

    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Drug Side Effect Prediction Report", styles["Title"]))
    elements.append(Spacer(1,12))
    elements.append(Paragraph(f"Description: {row[1]}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {row[2]}", styles["Normal"]))
    elements.append(Paragraph(f"Gender: {row[3]}", styles["Normal"]))
    elements.append(Paragraph(f"Dosage: {row[4]}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {row[5]}", styles["Normal"]))

    doc.build(elements)

    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False, threaded=False)
