# app.py
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from collections import Counter
from functools import wraps
from datetime import datetime

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
app.secret_key = 'your_very_secret_random_key_for_sessions_emr_project_v6' # IMPORTANT: CHANGE THIS!
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
print(f"--- Expected data directory: {os.path.abspath(DATA_DIR)} ---")

# --- Mock User Store ---
users = {
    "admin": {
        "password_hash": generate_password_hash("password123"),
        "name": "Admin User",
        "role": "administrator"
    },
    "doctor": {
        "password_hash": generate_password_hash("docpass"),
        "name": "Dr. Smith",
        "role": "doctor"
    }
}

# --- Helper Function to Convert NaT to None in DataFrame columns ---
def convert_df_dates_to_none(df, date_column_names):
    """
    Converts NaT in specified datetime columns of a DataFrame to None.
    It ensures that Python's None is used by changing column dtype to 'object' if NaTs were present.
    """
    if df.empty:
        return df
    for col_name in date_column_names:
        if col_name in df.columns and pd.api.types.is_datetime64_any_dtype(df[col_name]):
            # Check if there are any NaT values in the column
            nat_mask = df[col_name].isnull() # pd.isnull() is True for pd.NaT
            if nat_mask.any():
                # Convert the column to object type FIRST. This allows it to hold Python's None.
                df[col_name] = df[col_name].astype(object)
                # Where the mask is True (it was NaT), assign Python's None.
                df.loc[nat_mask, col_name] = None
    return df

# --- Helper Functions to Load Data ---
def load_data(filename):
    """Loads a CSV file into a Pandas DataFrame with improved date handling."""
    file_path = os.path.join(DATA_DIR, filename)
    print(f"Attempting to load: {os.path.abspath(file_path)}") # For debugging file paths
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist at path: {os.path.abspath(file_path)}")
        flash(f"Data file '{filename}' not found. Some information may be missing.", "danger")
        return pd.DataFrame()
    try:
        # Define which string values should be considered NA for specific columns during CSV read
        na_values_map = {'end_date': ['']} 
        temp_df_check = pd.read_csv(file_path, nrows=0) # Read only header to check columns
        valid_na_values = {k: v for k, v in na_values_map.items() if k in temp_df_check.columns}
        
        df = pd.read_csv(file_path, na_values=valid_na_values, keep_default_na=True)

        # Define all columns that are expected to contain date information
        date_columns_to_parse = ['encounter_date', 'date_of_birth', 'start_date', 'end_date', 'timestamp']
        for col in date_columns_to_parse:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce') # Convert to datetime, errors become NaT
        
        # After parsing, convert any resulting NaT values to Python's None for template compatibility
        df = convert_df_dates_to_none(df, date_columns_to_parse)

        print(f"Successfully loaded and processed: {filename}")
        return df
    except Exception as e: 
        print(f"ERROR: Exception while loading or processing {filename}: {type(e).__name__} - {e}")
        flash(f"Error loading/processing data from '{filename}'. Details: {str(e)}", "danger")
        return pd.DataFrame()

# --- Decorators ---
def login_required(f):
    """Decorator to ensure a user is logged in to access a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: # Check if user_id is stored in session
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url)) # Redirect to login, preserving intended page
        return f(*args, **kwargs)
    return decorated_function

# --- Routes and View Functions ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if 'user_id' in session: # If already logged in, redirect to dashboard
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_data = users.get(username) # Check against mock user store

        if user_data and check_password_hash(user_data['password_hash'], password):
            # Store user info in session upon successful login
            session['user_id'] = username
            session['user_name'] = user_data['name']
            session['user_role'] = user_data['role']
            flash(f"Welcome back, {user_data['name']}!", 'success')
            next_page = request.args.get('next') # For redirecting after login
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required # Ensure only logged-in users can access logout
def logout():
    """Logs the user out by clearing the session."""
    session.clear() # Remove all items from the session
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required # Protect the dashboard
def index():
    """View function for the EMR dashboard, including filter logic."""
    print("\n--- Loading data for Dashboard ---")
    # Load all necessary dataframes
    all_patients_df = load_data('patients.csv')
    all_encounters_df = load_data('encounters.csv')
    all_diagnoses_df = load_data('diagnoses.csv')
    all_medications_df = load_data('medications.csv')

    # Get filter values from URL query parameters (e.g., /?filter_year=2023)
    filter_year_str = request.args.get('filter_year', type=str)
    filter_encounter_type = request.args.get('filter_encounter_type', type=str)

    # --- Apply Filters ---
    # Start with copies of the original DataFrames to apply filters iteratively
    filtered_encounters_df = all_encounters_df.copy()
    
    # Apply Year Filter to encounters
    if filter_year_str and filter_year_str.isdigit():
        filter_year = int(filter_year_str)
        if 'encounter_date' in filtered_encounters_df.columns and not filtered_encounters_df.empty:
            # Ensure 'encounter_date' is not None/NaT before accessing .dt.year
            valid_year_encounters = filtered_encounters_df.dropna(subset=['encounter_date'])
            filtered_encounters_df = valid_year_encounters[valid_year_encounters['encounter_date'].dt.year == filter_year]
    
    # Apply Encounter Type Filter to the (potentially) year-filtered encounters
    if filter_encounter_type and 'encounter_type' in filtered_encounters_df.columns and not filtered_encounters_df.empty:
        filtered_encounters_df = filtered_encounters_df[filtered_encounters_df['encounter_type'] == filter_encounter_type]

    # Determine which patients and diagnoses are relevant based on filtered encounters
    if (filter_year_str or filter_encounter_type): # If any encounter filter was applied
        if not filtered_encounters_df.empty:
            patient_ids_from_filtered_encounters = filtered_encounters_df['patient_id'].unique()
            filtered_patients_df = all_patients_df[all_patients_df['patient_id'].isin(patient_ids_from_filtered_encounters)]
            
            encounter_ids_for_diagnoses = filtered_encounters_df['encounter_id'].unique()
            filtered_diagnoses_df = all_diagnoses_df[all_diagnoses_df['encounter_id'].isin(encounter_ids_for_diagnoses)]
        else: # Filters were applied but resulted in no matching encounters
            filtered_patients_df = pd.DataFrame(columns=all_patients_df.columns if not all_patients_df.empty else []) 
            filtered_diagnoses_df = pd.DataFrame(columns=all_diagnoses_df.columns if not all_diagnoses_df.empty else [])
    else: # No encounter filters applied, use all patients and diagnoses
        filtered_patients_df = all_patients_df.copy()
        filtered_diagnoses_df = all_diagnoses_df.copy()


    # --- Calculate Dashboard Metrics using FILTERED data ---
    patient_count = len(filtered_patients_df) if not filtered_patients_df.empty else 0
    encounter_count = len(filtered_encounters_df) if not filtered_encounters_df.empty else 0
    
    # Active medications count is typically overall, not filtered by encounter context here
    active_medications_count = 0
    if not all_medications_df.empty and 'end_date' in all_medications_df.columns:
        active_medications_count = all_medications_df['end_date'].isnull().sum()

    # Recent Encounters from the filtered set
    recent_encounters = []
    if not filtered_encounters_df.empty and 'encounter_date' in filtered_encounters_df.columns:
        valid_recent_encounters = filtered_encounters_df.dropna(subset=['encounter_date'])
        if not valid_recent_encounters.empty:
            recent_encounters_df = valid_recent_encounters.sort_values(by='encounter_date', ascending=False).head(5)
            recent_encounters = recent_encounters_df.to_dict('records')

    # Common Diagnoses from the filtered set
    common_diagnoses_list = []
    if not filtered_diagnoses_df.empty and 'diagnosis_description' in filtered_diagnoses_df.columns:
        diagnosis_counts = Counter(filtered_diagnoses_df['diagnosis_description'].dropna())
        common_diagnoses_list = diagnosis_counts.most_common(5)

    # --- Data for Charts using FILTERED data ---
    # Gender Distribution (from filtered patients)
    gender_data = {"labels": [], "counts": []}
    if not filtered_patients_df.empty and 'gender' in filtered_patients_df.columns:
        gender_counts = filtered_patients_df['gender'].value_counts()
        gender_data['labels'] = gender_counts.index.tolist()
        gender_data['counts'] = gender_counts.values.tolist()

    # Age Group Distribution (from filtered patients)
    age_group_data = {"labels": [], "counts": []}
    if not filtered_patients_df.empty and 'date_of_birth' in filtered_patients_df.columns:
        now = datetime.now()
        valid_dob_patients = filtered_patients_df.dropna(subset=['date_of_birth']) # Ensure DOB is not None
        if not valid_dob_patients.empty:
            ages = valid_dob_patients['date_of_birth'].apply(
                lambda x: now.year - x.year - ((now.month, now.day) < (x.month, x.day)) # Age calculation
            )
            ages = ages[ages >= 0] # Filter out potential negative ages if data is inconsistent
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
            labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
            if not ages.empty: # Check if ages Series is not empty before pd.cut
                age_groups = pd.cut(ages, bins=bins, labels=labels, right=True, include_lowest=True)
                age_group_counts = age_groups.value_counts().sort_index()
                age_group_data['labels'] = age_group_counts.index.tolist()
                age_group_data['counts'] = age_group_counts.values.tolist()

    # Encounter Types (from filtered encounters)
    encounter_type_data = {"labels": [], "counts": []}
    if not filtered_encounters_df.empty and 'encounter_type' in filtered_encounters_df.columns:
        encounter_type_counts = filtered_encounters_df['encounter_type'].value_counts()
        encounter_type_data['labels'] = encounter_type_counts.index.tolist()
        encounter_type_data['counts'] = encounter_type_counts.values.tolist()
    
    print("--- Finished loading data for Dashboard ---")
    return render_template(
        'dashboard.html',
        patient_count=patient_count,
        encounter_count=encounter_count,
        active_medications_count=active_medications_count,
        common_diagnoses=common_diagnoses_list,
        recent_encounters=recent_encounters,
        gender_data=gender_data,
        age_group_data=age_group_data,
        encounter_type_data=encounter_type_data
    )

@app.route('/patients')
@login_required
def patient_list():
    """View function for the patient listing page."""
    print("\n--- Loading data for Patient List ---")
    patients_df = load_data('patients.csv') 
    patients_data = []
    if not patients_df.empty:
        patients_data = patients_df.to_dict('records') # load_data handles date conversions
    print("--- Finished loading data for Patient List ---")
    return render_template('patient_list.html', patients=patients_data)

@app.route('/patient/<patient_id>')
@login_required
def patient_detail(patient_id):
    """View function for displaying details of a single patient."""
    print(f"\n--- Loading data for Patient Detail: {patient_id} ---")
    # load_data ensures date columns have NaT converted to None
    patients_df = load_data('patients.csv')
    encounters_df = load_data('encounters.csv')
    diagnoses_df = load_data('diagnoses.csv')
    medications_df = load_data('medications.csv')
    vitals_df = load_data('vitals.csv')

    patient_info = None
    patient_encounters = []
    patient_diagnoses = []
    patient_medications = []
    patient_vitals = []

    if patients_df.empty: # Check if core patient data loaded
        flash("Core patient data file could not be loaded. Cannot display patient details.", "danger")
        return redirect(url_for('patient_list')) # Redirect if patients.csv is missing/empty

    patient_series_df = patients_df[patients_df['patient_id'] == patient_id] # Filter for the patient
    if not patient_series_df.empty:
        patient_info = patient_series_df.iloc[0].to_dict() # load_data ensures dates are None if NaT
    else:
        flash(f"Patient with ID {patient_id} not found.", "warning")
        return redirect(url_for('patient_list')) # Redirect if specific patient not found

    # Get encounters for this patient
    if not encounters_df.empty and patient_id:
        patient_encounters_df = encounters_df[encounters_df['patient_id'] == patient_id]
        # Sort by 'encounter_date' only if it's not all None/NaT
        if 'encounter_date' in patient_encounters_df.columns and not patient_encounters_df['encounter_date'].isnull().all():
            patient_encounters = patient_encounters_df.sort_values(by='encounter_date', ascending=False).to_dict('records')
        else:
            patient_encounters = patient_encounters_df.to_dict('records')


        # Get diagnoses linked to these encounters
        if not diagnoses_df.empty and patient_encounters: # Only if encounters were found
            patient_encounter_ids = [enc['encounter_id'] for enc in patient_encounters if enc.get('encounter_id')] # Get list of encounter IDs
            if patient_encounter_ids: # Only if there are encounter IDs
                patient_diagnoses_df = diagnoses_df[diagnoses_df['encounter_id'].isin(patient_encounter_ids)]
                patient_diagnoses = patient_diagnoses_df.to_dict('records')

    # Get medications for this patient
    if not medications_df.empty and patient_id:
        patient_medications_df = medications_df[medications_df['patient_id'] == patient_id]
        if 'start_date' in patient_medications_df.columns and not patient_medications_df['start_date'].isnull().all():
            patient_medications = patient_medications_df.sort_values(by='start_date', ascending=False).to_dict('records')
        else:
            patient_medications = patient_medications_df.to_dict('records')

    # Get vitals for this patient
    if not vitals_df.empty and patient_id:
        patient_vitals_df = vitals_df[vitals_df['patient_id'] == patient_id]
        if 'timestamp' in patient_vitals_df.columns and not patient_vitals_df['timestamp'].isnull().all():
            patient_vitals = patient_vitals_df.sort_values(by='timestamp', ascending=False).to_dict('records')
        else:
            patient_vitals = patient_vitals_df.to_dict('records')
    
    print(f"--- Finished loading data for Patient Detail: {patient_id} ---")
    return render_template(
        'patient_detail.html',
        patient=patient_info,
        encounters=patient_encounters,
        diagnoses=patient_diagnoses,
        medications=patient_medications,
        vitals=patient_vitals
    )

# --- Main execution point ---
if __name__ == '__main__':
    # Runs the Flask development server
    # Debug=True allows for automatic reloading when code changes and provides detailed error pages
    app.run(debug=True)