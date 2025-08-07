# app.py - Complete Database Solution with All Routes
import os
import io
import json
import pandas as pd
import sqlite3
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import traceback
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Use SQLite database for persistent storage
DATABASE_URL = os.environ.get('DATABASE_URL', 'simulation_data.db')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# A mapping of possible column names from uploaded files to the standardized database field names.
# This helps automatically handle minor variations or typos in column headers.
COLUMN_MAPPING = {
    'date': 'Date', 'session_date': 'Date',
    'branch': 'Branch', 'location': 'Branch',
    'operator': 'Operator',
    'start_time': 'Start_Time', 'session_start': 'Start_Time',
    'length_hours': 'Length_Hours', 'session_length': 'Length_Hours', 'length': 'Length_Hours',
    'patient_name': 'Patient_Name', 'recording_name': 'Patient_Name',
    'room': 'Room', 'sim_room': 'Room',
    'debrief_location': 'Debrief_Location',
    'other_locations': 'Other_Locations',
    'manikin': 'Manikin', 'mannequin': 'Manikin',
    'participants': 'Participants', 'num_participants': 'Participants',
    'department': 'Department', 'dept': 'Department',
    'course': 'Course', 'nursing_course': 'Course',
    'simulated_participants': 'Simulated_Participants', 'sp_used': 'Simulated_Participants',
    'num_simulated_participants': 'Num_Simulated_Participants', 'sp_number': 'Num_Simulated_Participants',
    'sp_name': 'SP_Name',
}

FIELDS = [
    'Date','Branch','Operator','Start_Time','Length_Hours','Patient_Name','Room',
    'Debrief_Location','Other_Locations','Manikin','Participants','Department',
    'Course','Simulated_Participants','Num_Simulated_Participants','SP_Name'
]

def init_database():
    """Initialize SQLite database with proper schema"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT,
                Branch TEXT,
                Operator TEXT,
                Start_Time TEXT,
                Length_Hours REAL,
                Patient_Name TEXT,
                Room TEXT,
                Debrief_Location TEXT,
                Other_Locations TEXT,
                Manikin TEXT,
                Participants INTEGER,
                Department TEXT,
                Course TEXT,
                Simulated_Participants TEXT,
                Num_Simulated_Participants INTEGER,
                SP_Name TEXT,
                Contact_Hours REAL,
                Week INTEGER,
                Month INTEGER,
                Quarter INTEGER,
                Year INTEGER,
                Week_Year TEXT,
                Month_Year TEXT,
                Quarter_Year TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON simulation_records(Date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_branch ON simulation_records(Branch)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_room ON simulation_records(Room)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_department ON simulation_records(Department)')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        raise

def save_record_to_db(record_data):
    """Save a single record to database"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        processed_record = process_single_record(record_data)
        
        placeholders = ', '.join(['?' for _ in range(len(processed_record))])
        columns = ', '.join(processed_record.keys())
        
        cursor.execute(f'''
            INSERT INTO simulation_records ({columns})
            VALUES ({placeholders})
        ''', list(processed_record.values()))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Record saved to database with ID: {record_id}")
        return record_id
        
    except Exception as e:
        print(f"❌ Error saving record to database: {e}")
        raise

def load_records_from_db(page=1, per_page=10, start_date=None, end_date=None, branch=None):
    """Load records from database with pagination and filtering"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        where_conditions = []
        params = []
        
        if start_date:
            where_conditions.append("Date >= ?")
            params.append(start_date)
            
        if end_date:
            where_conditions.append("Date <= ?")
            params.append(end_date)
            
        if branch:
            where_conditions.append("Branch LIKE ?")
            params.append(f"%{branch}%")
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        count_query = f"SELECT COUNT(*) FROM simulation_records {where_clause}"
        cursor.execute(count_query, params)
        total_records = cursor.fetchone()[0]
        
        offset = (page - 1) * per_page
        query = f'''
            SELECT * FROM simulation_records 
            {where_clause}
            ORDER BY Date DESC, id DESC
            LIMIT ? OFFSET ?
        '''
        
        cursor.execute(query, params + [per_page, offset])
        records = cursor.fetchall()
        
        records_list = [dict(row) for row in records]
        conn.close()
        
        total_pages = max(1, (total_records + per_page - 1) // per_page)
        
        pagination = {
            'page': page,
            'per_page': per_page,
            'total_records': total_records,
            'total_pages': total_pages,
            'has_prev': page > 1,
            'has_next': page < total_pages,
            'start_record': offset + 1 if total_records > 0 else 0,
            'end_record': min(offset + per_page, total_records)
        }
        
        return records_list, pagination
        
    except Exception as e:
        print(f"❌ Error loading records from database: {e}")
        raise

def get_all_records_from_db():
    """Get all records from database for analytics and bulk operations"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM simulation_records ORDER BY Date DESC, id DESC')
        records = cursor.fetchall()
        
        records_list = [dict(row) for row in records]
        conn.close()
        
        return records_list
        
    except Exception as e:
        print(f"❌ Error loading all records from database: {e}")
        return []

def update_record_in_db(record_id, record_data):
    """Update a record in database"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        processed_record = process_single_record(record_data)
        processed_record['updated_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{key} = ?" for key in processed_record.keys()])
        
        cursor.execute(f'''
            UPDATE simulation_records 
            SET {set_clause}
            WHERE id = ?
        ''', list(processed_record.values()) + [record_id])
        
        conn.commit()
        conn.close()
        
        print(f"✅ Record {record_id} updated in database")
        return True
        
    except Exception as e:
        print(f"❌ Error updating record in database: {e}")
        raise

def delete_record_from_db(record_id):
    """Delete a record from database"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM simulation_records WHERE id = ?', (record_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Record {record_id} deleted from database")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting record from database: {e}")
        raise

def bulk_delete_records_from_db(record_ids):
    """Delete multiple records from database"""
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        placeholders = ', '.join(['?' for _ in record_ids])
        cursor.execute(f'DELETE FROM simulation_records WHERE id IN ({placeholders})', record_ids)
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ {deleted_count} records deleted from database")
        return deleted_count
        
    except Exception as e:
        return jsonify({'error': 'Error bulk deleting records', 'message': str(e)}), 500

def process_single_record(record_data):
    """Process a single record for database storage"""
    processed = {}
    
    for field in FIELDS:
        processed[field] = record_data.get(field, '')
    
    if processed['Date']:
        try:
            processed['Date'] = pd.to_datetime(processed['Date']).strftime('%Y-%m-%d')
        except:
            processed['Date'] = ''
    
    processed['Participants'] = float(record_data.get('Participants', 0)) or 0
    processed['Length_Hours'] = float(record_data.get('Length_Hours', 0)) or 0.0
    processed['Num_Simulated_Participants'] = int(record_data.get('Num_Simulated_Participants', 0)) or 0
    
    processed['Contact_Hours'] = processed['Participants'] * processed['Length_Hours']
    
    if processed['Date']:
        try:
            date_obj = pd.to_datetime(processed['Date'])
            processed['Week'] = int(date_obj.isocalendar().week)
            processed['Month'] = int(date_obj.month)
            processed['Quarter'] = int(date_obj.quarter)
            processed['Year'] = int(date_obj.year)
            processed['Week_Year'] = f"{processed['Year']}-W{processed['Week']:02d}"
            processed['Month_Year'] = date_obj.strftime('%Y-%m')
            processed['Quarter_Year'] = f"{processed['Year']}-Q{processed['Quarter']}"
        except:
            processed['Week'] = processed['Month'] = processed['Quarter'] = processed['Year'] = None
            processed['Week_Year'] = processed['Month_Year'] = processed['Quarter_Year'] = ''
    
    return processed

def standardize_columns(df, mapping):
    """Standardize column names of a DataFrame based on a mapping dictionary."""
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    df.columns = df.columns.str.lower()
    
    renamed_columns = {col: mapping.get(col, col) for col in df.columns}
    
    return df.rename(columns=renamed_columns)


# Routes
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/records')
def records():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        search_branch = request.args.get('search_branch', '')
        
        records_list, pagination = load_records_from_db(
            page=page, per_page=per_page, 
            start_date=start_date, end_date=end_date, 
            branch=search_branch
        )
        
        return jsonify({
            'records': records_list,
            'pagination': pagination
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'records': [],
            'pagination': {
                'page': 1, 'per_page': 10, 'total_records': 0,
                'total_pages': 0, 'has_prev': False, 'has_next': False,
                'start_record': 0, 'end_record': 0
            }
        }), 500

@app.route('/add_record', methods=['POST'])
def add_record():
    try:
        record = {field: request.form.get(field, '') for field in FIELDS}
        
        if 'Other_Locations' in request.form:
            other_locations = request.form.getlist('Other_Locations')
            record['Other_Locations'] = ', '.join(other_locations)
        
        if 'Operator' in request.form:
            operators = request.form.getlist('Operator')
            record['Operator'] = ', '.join(operators)
        
        save_record_to_db(record)
        return redirect(url_for('home'))
        
    except Exception as e:
        return jsonify({'error': 'Error adding record', 'message': str(e)}), 500

@app.route('/update_record', methods=['POST'])
def update_record():
    try:
        record_id = int(request.form.get('update_index', -1))
        
        if record_id > 0:
            record = {field: request.form.get(field, '') for field in FIELDS}
            
            if 'Other_Locations' in request.form:
                other_locations = request.form.getlist('Other_Locations')
                record['Other_Locations'] = ', '.join(other_locations)
            
            if 'Operator' in request.form:
                operators = request.form.getlist('Operator')
                record['Operator'] = ', '.join(operators)
            
            update_record_in_db(record_id, record)
        
        return redirect(url_for('home'))
        
    except Exception as e:
        return jsonify({'error': 'Error updating record', 'message': str(e)}), 500

@app.route('/delete_record', methods=['POST'])
def delete_record():
    try:
        record_id = int(request.form['index'])
        delete_record_from_db(record_id)
        return redirect(url_for('home'))
        
    except Exception as e:
        return jsonify({'error': 'Error deleting record', 'message': str(e)}), 500

@app.route('/bulk_delete', methods=['POST'])
def bulk_delete():
    try:
        selected_ids = request.json.get('indices', [])
        
        if not selected_ids:
            return jsonify({'error': 'No records selected'}), 400
        
        deleted_count = bulk_delete_records_from_db(selected_ids)
        return jsonify({'message': f'Successfully deleted {deleted_count} records'}), 200
        
    except Exception as e:
        return jsonify({'error': 'Error deleting records', 'message': str(e)}), 500

@app.route('/bulk_download', methods=['POST'])
def bulk_download():
    try:
        selected_indices = request.json.get('indices', [])
        
        if not selected_indices:
            return jsonify({'error': 'No records selected'}), 400
        
        conn = sqlite3.connect(DATABASE_URL)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        placeholders = ', '.join(['?' for _ in selected_indices])
        cursor.execute(f'SELECT * FROM simulation_records WHERE id IN ({placeholders})', selected_indices)
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return jsonify({'error': 'No valid data to download'}), 400
        
        df = pd.DataFrame([dict(row) for row in records])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Selected Records', index=False)
        
        output.seek(0)
        return send_file(
            output,
            download_name=f'selected_simulation_data_{datetime.now().strftime("%Y%m%d")}.xlsx',
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': 'Error downloading records', 'message': str(e)}), 500

@app.route('/download')
def download():
    try:
        records = get_all_records_from_db()
        
        if not records:
            return jsonify({'error': 'No data to download'}), 400
        
        df = pd.DataFrame(records)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='All Data', index=False)
            
            if not df.empty:
                room_summary = df.groupby('Room').agg({
                    'Length_Hours': 'sum',
                    'Participants': 'sum', 
                    'Contact_Hours': 'sum'
                }).reset_index()
                room_summary.to_excel(writer, sheet_name='Room Summary', index=False)
        
        output.seek(0)
        return send_file(
            output,
            download_name=f'simulation_data_{datetime.now().strftime("%Y%m%d")}.xlsx',
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': 'Error downloading data', 'message': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, engine='openpyxl')
        
        # Standardize the column names
        df = standardize_columns(df, COLUMN_MAPPING)

        records_added = 0
        for _, row in df.iterrows():
            record_data = {field: row.get(field, '') for field in FIELDS}
            # Only save records with a valid date and at least a few key fields
            if record_data['Date'] and record_data['Branch'] and record_data['Room'] and record_data['Participants']:
                save_record_to_db(record_data)
                records_added += 1
        
        os.remove(filepath)
        
        return jsonify({'message': 'File uploaded successfully', 'rows_added': records_added}), 200
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

# Analytics API endpoints
@app.route('/api/summary_stats')
def api_summary_stats():
    try:
        records = get_all_records_from_db()
        
        if not records:
            return jsonify({
                'total_sessions': 0,
                'total_participants': 0,
                'total_contact_hours': 0,
                'avg_session_length': 0,
                'unique_rooms': 0,
                'unique_operators': 0
            })
        
        df = pd.DataFrame(records)
        
        df['Participants'] = pd.to_numeric(df['Participants'], errors='coerce').fillna(0)
        df['Contact_Hours'] = pd.to_numeric(df['Contact_Hours'], errors='coerce').fillna(0)
        df['Length_Hours'] = pd.to_numeric(df['Length_Hours'], errors='coerce').fillna(0)
        
        summary = {
            'total_sessions': int(len(df)),
            'total_participants': int(df['Participants'].sum()),
            'total_contact_hours': float(df['Contact_Hours'].sum()),
            'avg_session_length': float(df['Length_Hours'].mean()) if len(df) > 0 else 0,
            'unique_rooms': int(df['Room'].nunique()),
            'unique_operators': int(df['Operator'].nunique())
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({
            'total_sessions': 0,
            'total_participants': 0,
            'total_contact_hours': 0,
            'avg_session_length': 0,
            'unique_rooms': 0,
            'unique_operators': 0,
            'error': str(e)
        })

@app.route('/api/room_utilization')
def api_room_utilization():
    try:
        records = get_all_records_from_db()
        
        if not records:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        df = pd.DataFrame(records)
        valid_data = df.dropna(subset=['Room']).copy()
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        valid_data['Length_Hours'] = pd.to_numeric(valid_data['Length_Hours'], errors='coerce').fillna(0)
        
        room_totals = valid_data.groupby('Room')['Length_Hours'].sum().reset_index()
        
        rooms = room_totals['Room'].tolist()
        hours = room_totals['Length_Hours'].tolist()
        
        fig = go.Figure(data=[
            go.Bar(
                x=rooms,
                y=hours,
                text=[f"{h:.1f}h" for h in hours],
                textposition='outside',
                marker_color='#2563eb'
            )
        ])
        
        fig.update_layout(
            title='Room Utilization - Total Hours by Room',
            xaxis_title='Simulation Room',
            yaxis_title='Total Hours',
            height=400,
            yaxis=dict(rangemode='tozero')
        )
        
        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return jsonify({
            'graph': graphJSON,
            'data': room_totals.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/contact_hours_timeline')
def api_contact_hours_timeline():
    try:
        records = get_all_records_from_db()
        
        if not records:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        df = pd.DataFrame(records)
        valid_data = df.dropna(subset=['Date']).copy()
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        valid_data['Date'] = pd.to_datetime(valid_data['Date'], errors='coerce')
        valid_data = valid_data.dropna(subset=['Date'])
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        
        time_data = valid_data.groupby('Date')['Contact_Hours'].sum().reset_index().sort_values('Date')
        
        dates = time_data['Date'].tolist()
        contact_hours = time_data['Contact_Hours'].tolist()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=contact_hours,
                mode='lines+markers',
                marker=dict(size=8, color='#2563eb'),
                line=dict(width=3, color='#2563eb')
            )
        ])
        
        fig.update_layout(
            title='Contact Hours Over Time',
            xaxis_title='Date',
            yaxis_title='Contact Hours',
            height=400,
            yaxis=dict(rangemode='tozero')
        )
        
        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return jsonify({
            'graph': graphJSON,
            'data': time_data.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/learner_stats')
def api_learner_stats():
    try:
        period = request.args.get('period', 'month')
        records = get_all_records_from_db()
        
        if not records:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        df = pd.DataFrame(records)
        valid_data = df.dropna(subset=['Date']).copy()
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        valid_data['Date'] = pd.to_datetime(valid_data['Date'], errors='coerce')
        valid_data = valid_data.dropna(subset=['Date'])
        
        valid_data['Participants'] = pd.to_numeric(valid_data['Participants'], errors='coerce').fillna(0)
        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        
        if period == 'week':
            group_col = 'Week_Year'
            title = 'Weekly Learner Statistics'
        elif period == 'quarter':
            group_col = 'Quarter_Year'
            title = 'Quarterly Learner Statistics'
        else:
            group_col = 'Month_Year'
            title = 'Monthly Learner Statistics'
        
        if group_col not in valid_data.columns:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        valid_data = valid_data.dropna(subset=[group_col])
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        stats = valid_data.groupby(group_col).agg({
            'Participants': 'sum',
            'Contact_Hours': 'sum'
        }).reset_index().sort_values(group_col)
        
        if stats.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        periods = stats[group_col].tolist()
        participants = stats['Participants'].tolist()
        contact_hours = stats['Contact_Hours'].tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Participants',
            x=periods,
            y=participants,
            marker_color='lightblue',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            name='Contact Hours',
            x=periods,
            y=contact_hours,
            yaxis='y2',
            mode='lines+markers',
            marker=dict(color='red', size=8),
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(title=f'{period.title()} Period'),
            yaxis=dict(title='Number of Participants', side='left', color='blue', rangemode='tozero'),
            yaxis2=dict(title='Contact Hours', side='right', overlaying='y', color='red', rangemode='tozero'),
            height=400,
            hovermode='x unified'
        )
        
        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return jsonify({
            'graph': graphJSON,
            'data': stats.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/branch_contact_hours')
def api_branch_contact_hours():
    try:
        records = get_all_records_from_db()
        
        if not records:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        df = pd.DataFrame(records)
        valid_data = df.dropna(subset=['Branch']).copy()
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        
        branch_contact_hours = valid_data.groupby('Branch')['Contact_Hours'].sum().reset_index()
        
        if branch_contact_hours.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        branches = branch_contact_hours['Branch'].tolist()
        contact_hours = branch_contact_hours['Contact_Hours'].tolist()
        
        fig = go.Figure(data=[
            go.Bar(
                x=branches,
                y=contact_hours,
                text=[f"{h:.1f}h" for h in contact_hours],
                textposition='outside',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(branches)]
            )
        ])
        
        fig.update_layout(
            title='Contact Hours by Branch',
            xaxis_title='Branch',
            yaxis_title='Total Contact Hours',
            height=400,
            yaxis=dict(rangemode='tozero')
        )
        
        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return jsonify({
            'graph': graphJSON,
            'data': branch_contact_hours.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/branch_hours_breakdown')
def api_branch_hours_breakdown():
    try:
        records = get_all_records_from_db()

        if not records:
            return jsonify({'graph': None, 'message': 'No data available'})

        df = pd.DataFrame(records)
        valid_data = df.dropna(subset=['Branch']).copy()

        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})

        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        valid_data['Length_Hours'] = pd.to_numeric(valid_data['Length_Hours'], errors='coerce').fillna(0)

        # Aggregate data by branch
        branch_summary = valid_data.groupby('Branch').agg({
            'Contact_Hours': 'sum',
            'Length_Hours': 'sum'
        }).reset_index()

        if branch_summary.empty:
            return jsonify({'graph': None, 'message': 'No data available'})

        branches = branch_summary['Branch'].tolist()
        contact_hours = branch_summary['Contact_Hours'].tolist()
        session_hours = branch_summary['Length_Hours'].tolist()

        # Create the stacked bar chart
        fig = go.Figure(data=[
            go.Bar(name='Contact Hours', x=branches, y=contact_hours, marker_color='#4ECDC4'),
            go.Bar(name='Session Hours', x=branches, y=session_hours, marker_color='#45B7D1')
        ])

        # Customize the chart layout
        fig.update_layout(
            barmode='group',
            title='Contact Hours vs. Session Hours by Branch',
            xaxis_title='Branch',
            yaxis_title='Total Hours',
            height=400,
            yaxis=dict(rangemode='tozero'),
            legend_title_text='Hour Type'
        )

        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

        return jsonify({
            'graph': graphJSON,
            'data': branch_summary.to_dict('records')
        })

    except Exception as e:
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database on startup
if __name__ == '__main__':
    try:
        print("\n=== Initializing database ===")
        init_database()
        
        print(f"\n=== Starting Flask app ===")
        
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        print(f"Traceback: {traceback.format_exc()}")
