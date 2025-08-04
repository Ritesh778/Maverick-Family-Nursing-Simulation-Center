# app.py - Final Fixed Version
import os
import io
import json
import pandas as pd
import pickle
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
app.config['DATA_FILE'] = 'simulation_data.pkl'  # File to persist data
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

FIELDS = [
    'Date','Branch','Operator','Start_Time','Length_Hours','Patient_Name','Room',
    'Debrief_Location','Other_Locations','Manikin','Participants','Department',
    'Course','Simulated_Participants','Num_Simulated_Participants','SP_Name'
]

# Load persisted data on startup
def load_data():
    """Load data from pickle file"""
    try:
        if os.path.exists(app.config['DATA_FILE']):
            with open(app.config['DATA_FILE'], 'rb') as f:
                data = pickle.load(f)
                df = pd.DataFrame(data) if data else pd.DataFrame()
                print(f"Loaded {len(df)} records from file")
                return df
    except Exception as e:
        print(f"Error loading data: {e}")
    return pd.DataFrame()

def save_data():
    """Save data to pickle file"""
    try:
        with open(app.config['DATA_FILE'], 'wb') as f:
            pickle.dump(data_store.to_dict('records'), f)
        print(f"Saved {len(data_store)} records to file")
    except Exception as e:
        print(f"Error saving data: {e}")

# Initialize data store with persisted data
data_store = load_data()

# Helper function to clean data for JSON serialization
def clean_for_json(value):
    """Clean individual values for JSON serialization"""
    if pd.isna(value) or value is None:
        return ''
    elif isinstance(value, (np.integer, int)):
        return int(value)
    elif isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return 0
        return float(value)
    elif isinstance(value, (np.bool_, bool)):
        return bool(value)
    else:
        return str(value)

def serialize_dataframe_for_json(df):
    """Convert DataFrame to JSON-serializable format"""
    if df.empty:
        return []
    
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            value = row[col]
            
            # Special handling for date columns
            if col == 'Date':
                if pd.notna(value):
                    try:
                        if isinstance(value, str):
                            # Try to parse string date
                            parsed_date = pd.to_datetime(value).strftime('%Y-%m-%d')
                            record[col] = parsed_date
                        else:
                            # Convert datetime to string
                            record[col] = pd.to_datetime(value).strftime('%Y-%m-%d')
                    except:
                        record[col] = str(value) if value else ''
                else:
                    record[col] = ''
            # Special handling for numeric columns
            elif col in ['Participants', 'Length_Hours', 'Contact_Hours', 'Num_Simulated_Participants']:
                try:
                    record[col] = float(value) if pd.notna(value) else 0
                except:
                    record[col] = 0
            # Clean other values
            else:
                record[col] = clean_for_json(value)
        
        records.append(record)
    
    return records

# Helper functions
def process_data(df):
    """Process and clean the dataframe"""
    if df.empty:
        print("DataFrame is empty, returning empty DataFrame")
        return df
    
    print("Processing data...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
        
    # Ensure Date is properly parsed
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure numeric fields are properly converted with explicit handling
    df['Participants'] = pd.to_numeric(df['Participants'], errors='coerce').fillna(0)
    df['Length_Hours'] = pd.to_numeric(df['Length_Hours'], errors='coerce').fillna(0.0)
    
    # Calculate contact hours - this is critical
    df['Contact_Hours'] = df['Participants'] * df['Length_Hours']
    
    # Add time-based grouping columns only for valid dates
    valid_dates = df['Date'].notna()
    
    if valid_dates.any():
        # Create integer-based time columns to avoid float issues
        df.loc[valid_dates, 'Week'] = df.loc[valid_dates, 'Date'].dt.isocalendar().week.astype(int)
        df.loc[valid_dates, 'Month'] = df.loc[valid_dates, 'Date'].dt.month.astype(int)
        df.loc[valid_dates, 'Quarter'] = df.loc[valid_dates, 'Date'].dt.quarter.astype(int)
        df.loc[valid_dates, 'Year'] = df.loc[valid_dates, 'Date'].dt.year.astype(int)
        
        # Create proper time period labels as strings
        df.loc[valid_dates, 'Week_Year'] = (df.loc[valid_dates, 'Year'].astype(str) + '-W' + 
                                           df.loc[valid_dates, 'Week'].astype(str).str.zfill(2))
        df.loc[valid_dates, 'Month_Year'] = df.loc[valid_dates, 'Date'].dt.strftime('%Y-%m')
        df.loc[valid_dates, 'Quarter_Year'] = (df.loc[valid_dates, 'Year'].astype(str) + '-Q' + 
                                              df.loc[valid_dates, 'Quarter'].astype(str))
    
    print(f"Processed data: {len(df)} records")
    return df

def get_room_utilization_data(period='month'):
    """Get room utilization data by period"""
    print(f"\n=== Getting room utilization data for period: {period} ===")
    
    if data_store.empty:
        print("Data store is empty")
        return pd.DataFrame()
    
    # Get valid data
    valid_data = data_store.dropna(subset=['Room']).copy()
    
    if valid_data.empty:
        print("No valid data after filtering")
        return pd.DataFrame()
    
    # Ensure numeric columns
    valid_data['Length_Hours'] = pd.to_numeric(valid_data['Length_Hours'], errors='coerce').fillna(0)
    valid_data['Participants'] = pd.to_numeric(valid_data['Participants'], errors='coerce').fillna(0)
    valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
    
    # Group by Room to get totals
    room_totals = valid_data.groupby('Room').agg({
        'Length_Hours': 'sum',
        'Participants': 'sum', 
        'Contact_Hours': 'sum'
    }).reset_index()
    
    return room_totals

def get_contact_hours_over_time():
    """Get contact hours over time"""
    print(f"\n=== Getting contact hours timeline data ===")
    
    if data_store.empty:
        print("Data store is empty")
        return pd.DataFrame()
    
    # Filter valid data
    valid_data = data_store.dropna(subset=['Date']).copy()
    
    if valid_data.empty:
        print("No valid data with dates")
        return pd.DataFrame()
    
    # Ensure numeric columns
    valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
    valid_data['Participants'] = pd.to_numeric(valid_data['Participants'], errors='coerce').fillna(0)
    
    # Group by date
    time_data = valid_data.groupby('Date').agg({
        'Contact_Hours': 'sum',
        'Participants': 'sum'
    }).reset_index().sort_values('Date')
    
    return time_data

# Routes
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/records')
def records():
    try:
        print("\n=== /records endpoint called ===")
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        search_branch = request.args.get('search_branch', '')
        
        print(f"Parameters: page={page}, per_page={per_page}, start_date={start_date}, end_date={end_date}, search_branch={search_branch}")
        print(f"Data store status: empty={data_store.empty}, length={len(data_store)}")
        
        # Start with all data
        filtered_data = data_store.copy() if not data_store.empty else pd.DataFrame()
        
        if filtered_data.empty:
            print("No data available")
            return jsonify({
                'records': [],
                'pagination': {
                    'page': 1,
                    'per_page': per_page,
                    'total_records': 0,
                    'total_pages': 0,
                    'has_prev': False,
                    'has_next': False,
                    'start_record': 0,
                    'end_record': 0
                }
            })
        
        # Apply date range filter if provided
        if start_date:
            try:
                start_date_obj = pd.to_datetime(start_date)
                filtered_data = filtered_data[filtered_data['Date'] >= start_date_obj]
                print(f"After start date filter: {len(filtered_data)} records")
            except Exception as e:
                print(f"Error parsing start date: {e}")
        
        if end_date:
            try:
                end_date_obj = pd.to_datetime(end_date)
                filtered_data = filtered_data[filtered_data['Date'] <= end_date_obj]
                print(f"After end date filter: {len(filtered_data)} records")
            except Exception as e:
                print(f"Error parsing end date: {e}")
        
        # Apply branch filter if provided
        if search_branch:
            try:
                filtered_data = filtered_data[filtered_data['Branch'].str.contains(search_branch, case=False, na=False)]
                print(f"After branch filter: {len(filtered_data)} records")
            except Exception as e:
                print(f"Error applying branch filter: {e}")
        
        # Calculate pagination
        total_records = len(filtered_data)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get paginated data
        if total_records > 0:
            paginated_data = filtered_data.iloc[start_idx:end_idx]
        else:
            paginated_data = pd.DataFrame()
        
        # Calculate pagination info
        total_pages = max(1, (total_records + per_page - 1) // per_page)
        has_prev = page > 1
        has_next = page < total_pages
        
        # Convert to JSON-serializable format
        records_list = serialize_dataframe_for_json(paginated_data)
        
        result = {
            'records': records_list,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_records': total_records,
                'total_pages': total_pages,
                'has_prev': has_prev,
                'has_next': has_next,
                'start_record': start_idx + 1 if total_records > 0 else 0,
                'end_record': min(end_idx, total_records)
            }
        }
        
        print(f"✅ Returning {len(records_list)} records successfully")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error in /records endpoint: {str(e)}"
        error_trace = traceback.format_exc()
        
        print(f"❌ {error_msg}")
        print(f"Traceback:\n{error_trace}")
        
        return jsonify({
            'error': error_msg,
            'message': str(e),
            'records': [],
            'pagination': {
                'page': 1,
                'per_page': 10,
                'total_records': 0,
                'total_pages': 0,
                'has_prev': False,
                'has_next': False,
                'start_record': 0,
                'end_record': 0
            }
        }), 500

@app.route('/add_record', methods=['POST'])
def add_record():
    try:
        global data_store
        
        print("\n=== Adding new record ===")
        
        record = {field: request.form.get(field, '') for field in FIELDS}
        
        # Handle multiple selections for Other_Locations
        if 'Other_Locations' in request.form:
            other_locations = request.form.getlist('Other_Locations')
            record['Other_Locations'] = ', '.join(other_locations)
        
        print("New record data:", record)
        
        new_row = pd.DataFrame([record])
        processed_row = process_data(new_row)
        
        if data_store.empty:
            data_store = processed_row.copy()
        else:
            data_store = pd.concat([data_store, processed_row], ignore_index=True)
        
        # Reprocess all data to ensure consistency
        data_store = process_data(data_store)
        
        print(f"Data store now has {len(data_store)} records")
        
        # Save data to file
        save_data()
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Error in add_record: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error adding record', 'message': str(e)}), 500

@app.route('/update_record', methods=['POST'])
def update_record():
    try:
        global data_store
        
        print("\n=== Updating record ===")
        
        # Get the index of the record to update
        update_index = int(request.form.get('update_index', -1))
        
        if 0 <= update_index < len(data_store):
            # Create updated record
            record = {field: request.form.get(field, '') for field in FIELDS}
            
            # Handle multiple selections for Other_Locations
            if 'Other_Locations' in request.form:
                other_locations = request.form.getlist('Other_Locations')
                record['Other_Locations'] = ', '.join(other_locations)
            
            # Process the updated record
            new_row = pd.DataFrame([record])
            processed_row = process_data(new_row)
            
            # Update the specific row
            for field in FIELDS:
                if field in data_store.columns:
                    data_store.iloc[update_index, data_store.columns.get_loc(field)] = processed_row.iloc[0][field]
            
            # Update calculated fields
            if 'Contact_Hours' in data_store.columns:
                data_store.iloc[update_index, data_store.columns.get_loc('Contact_Hours')] = processed_row.iloc[0]['Contact_Hours']
            
            # Reprocess all data to ensure consistency
            data_store = process_data(data_store)
            
            print(f"Updated data store with {len(data_store)} records")
            
            # Save data to file
            save_data()
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Error in update_record: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error updating record', 'message': str(e)}), 500

@app.route('/delete_record', methods=['POST'])
def delete_record():
    try:
        global data_store
        index = int(request.form['index'])
        if 0 <= index < len(data_store):
            print(f"\n=== Deleting record at index {index} ===")
            data_store.drop(data_store.index[index], inplace=True)
            data_store.reset_index(drop=True, inplace=True)
            
            # Reprocess all data to ensure consistency
            if not data_store.empty:
                data_store = process_data(data_store)
            
            print(f"Data store now has {len(data_store)} records")
            
            # Save data to file
            save_data()
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Error in delete_record: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error deleting record', 'message': str(e)}), 500

@app.route('/bulk_delete', methods=['POST'])
def bulk_delete():
    try:
        global data_store
        
        print("\n=== Bulk deleting records ===")
        
        # Get selected indices from request
        selected_indices = request.json.get('indices', [])
        
        if not selected_indices:
            return jsonify({'error': 'No records selected'}), 400
        
        # Sort indices in descending order to avoid index shifting issues
        selected_indices.sort(reverse=True)
        
        # Delete records
        for index in selected_indices:
            if 0 <= index < len(data_store):
                data_store.drop(data_store.index[index], inplace=True)
        
        # Reset index
        data_store.reset_index(drop=True, inplace=True)
        
        # Reprocess all data to ensure consistency
        if not data_store.empty:
            data_store = process_data(data_store)
        
        print(f"Data store now has {len(data_store)} records after bulk delete")
        
        # Save data to file
        save_data()
        
        return jsonify({'message': f'Successfully deleted {len(selected_indices)} records'}), 200
        
    except Exception as e:
        print(f"Error in bulk delete: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error deleting records', 'message': str(e)}), 500

@app.route('/bulk_download', methods=['POST'])
def bulk_download():
    try:
        print("\n=== Bulk downloading records ===")
        
        # Get selected indices from request
        selected_indices = request.json.get('indices', [])
        
        if not selected_indices:
            return jsonify({'error': 'No records selected'}), 400
        
        # Get selected records
        if data_store.empty:
            return jsonify({'error': 'No data available'}), 400
            
        selected_data = data_store.iloc[selected_indices]
        
        if selected_data.empty:
            return jsonify({'error': 'No valid data to download'}), 400
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            selected_data.to_excel(writer, sheet_name='Selected Records', index=False)
        
        output.seek(0)
        return send_file(
            output,
            download_name=f'selected_simulation_data_{datetime.now().strftime("%Y%m%d")}.xlsx',
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        print(f"Error in bulk download: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error downloading records', 'message': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        global data_store
        
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

        # Load the file
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, engine='openpyxl')
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Process and add to data store
        processed_df = process_data(df)
        
        if data_store.empty:
            data_store = processed_df.copy()
        else:
            data_store = pd.concat([data_store, processed_df], ignore_index=True)
        
        # Reprocess all data to ensure consistency
        data_store = process_data(data_store)
        
        # Save data to file
        save_data()
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({'message': 'File uploaded successfully', 'rows_added': len(df)}), 200
        
    except Exception as e:
        print(f"Error in upload: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/download')
def download():
    try:
        if data_store.empty:
            return jsonify({'error': 'No data to download'}), 400
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main data
            data_store.to_excel(writer, sheet_name='All Data', index=False)
            
            # Room utilization for different periods
            for period in ['month', 'week', 'quarter']:
                room_util = get_room_utilization_data(period)
                if not room_util.empty:
                    room_util.to_excel(writer, sheet_name=f'Room Util {period.title()}', index=False)
        
        output.seek(0)
        return send_file(
            output,
            download_name=f'simulation_data_{datetime.now().strftime("%Y%m%d")}.xlsx',
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        print(f"Error in download: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Error downloading data', 'message': str(e)}), 500

# Visualization API routes
@app.route('/api/room_utilization')
def api_room_utilization():
    """API endpoint for room utilization visualization"""
    try:
        period = request.args.get('period', 'month')
        data = get_room_utilization_data(period)
        
        if data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Create simple bar chart with explicit values
        rooms = data['Room'].tolist()
        hours = data['Length_Hours'].tolist()
        
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
            'data': serialize_dataframe_for_json(data)
        })
        
    except Exception as e:
        print(f"Error in room utilization API: {e}")
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/contact_hours_timeline')
def api_contact_hours_timeline():
    """API endpoint for contact hours over time"""
    try:
        data = get_contact_hours_over_time()
        
        if data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Create simple line chart with explicit values
        dates = data['Date'].tolist()
        contact_hours = data['Contact_Hours'].tolist()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=contact_hours,
                mode='lines+markers',
                marker=dict(size=8, color='#2563eb'),
                line=dict(width=3, color='#2563eb'),
                text=[f"{h:.1f}" for h in contact_hours],
                textposition='top center'
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
            'data': serialize_dataframe_for_json(data)
        })
        
    except Exception as e:
        print(f"Error in contact hours timeline API: {e}")
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/learner_stats')
def api_learner_stats():
    """API endpoint for learner statistics visualization"""
    try:
        period = request.args.get('period', 'month')
        
        if data_store.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Get valid data
        valid_data = data_store.dropna(subset=['Date']).copy()
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Ensure numeric columns
        valid_data['Participants'] = pd.to_numeric(valid_data['Participants'], errors='coerce').fillna(0)
        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        
        # Determine grouping column
        if period == 'week':
            group_col = 'Week_Year'
            title = 'Weekly Learner Statistics'
        elif period == 'quarter':
            group_col = 'Quarter_Year'
            title = 'Quarterly Learner Statistics'
        else:
            group_col = 'Month_Year'
            title = 'Monthly Learner Statistics'
        
        # Check if the grouping column exists and has data
        if group_col not in valid_data.columns:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Filter out rows where the grouping column is null
        valid_data = valid_data.dropna(subset=[group_col])
        
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Group data
        stats = valid_data.groupby(group_col).agg({
            'Participants': 'sum',
            'Contact_Hours': 'sum'
        }).reset_index()
        
        if stats.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Sort by time period
        stats = stats.sort_values(group_col)
        
        # Extract data for chart
        periods = stats[group_col].tolist()
        participants = stats['Participants'].tolist()
        contact_hours = stats['Contact_Hours'].tolist()
        
        # Create dual-axis chart
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
            yaxis=dict(
                title='Number of Participants',
                side='left',
                color='blue',
                rangemode='tozero'
            ),
            yaxis2=dict(
                title='Contact Hours',
                side='right',
                overlaying='y',
                color='red',
                rangemode='tozero'
            ),
            height=400,
            hovermode='x unified'
        )
        
        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return jsonify({
            'graph': graphJSON,
            'data': serialize_dataframe_for_json(stats)
        })
        
    except Exception as e:
        print(f"Error in learner stats API: {e}")
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/branch_contact_hours')
def api_branch_contact_hours():
    """API endpoint for branch contact hours visualization"""
    try:
        if data_store.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Get valid data
        valid_data = data_store.dropna(subset=['Branch']).copy()
        if valid_data.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Ensure numeric columns
        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        
        # Group by Branch and sum contact hours
        branch_contact_hours = valid_data.groupby('Branch')['Contact_Hours'].sum().reset_index()
        
        if branch_contact_hours.empty:
            return jsonify({'graph': None, 'message': 'No data available'})
        
        # Create simple bar chart
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
            'data': serialize_dataframe_for_json(branch_contact_hours)
        })
        
    except Exception as e:
        print(f"Error in branch contact hours API: {e}")
        return jsonify({'graph': None, 'message': 'Error loading chart', 'error': str(e)})

@app.route('/api/summary_stats')
def api_summary_stats():
    """API endpoint for summary statistics"""
    try:
        if data_store.empty:
            return jsonify({
                'total_sessions': 0,
                'total_participants': 0,
                'total_contact_hours': 0,
                'avg_session_length': 0,
                'unique_rooms': 0,
                'unique_operators': 0
            })
        
        # Use all data for summary
        valid_data = data_store.copy()
        
        # Ensure numeric columns
        valid_data['Participants'] = pd.to_numeric(valid_data['Participants'], errors='coerce').fillna(0)
        valid_data['Contact_Hours'] = pd.to_numeric(valid_data['Contact_Hours'], errors='coerce').fillna(0)
        valid_data['Length_Hours'] = pd.to_numeric(valid_data['Length_Hours'], errors='coerce').fillna(0)
        
        summary = {
            'total_sessions': int(len(valid_data)),
            'total_participants': int(valid_data['Participants'].sum()),
            'total_contact_hours': float(valid_data['Contact_Hours'].sum()),
            'avg_session_length': float(valid_data['Length_Hours'].mean()) if len(valid_data) > 0 else 0,
            'unique_rooms': int(valid_data['Room'].nunique()),
            'unique_operators': int(valid_data['Operator'].nunique())
        }
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"Error in summary stats API: {e}")
        return jsonify({
            'total_sessions': 0,
            'total_participants': 0,
            'total_contact_hours': 0,
            'avg_session_length': 0,
            'unique_rooms': 0,
            'unique_operators': 0,
            'error': str(e)
        })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Process existing data on startup
        if not data_store.empty:
            print("\n=== Processing existing data on startup ===")
            data_store = process_data(data_store)
            save_data()
        
        print(f"\n=== Starting Flask app with {len(data_store)} records ===")
        if not data_store.empty:
            print("Sample data in store:")
            print(data_store[['Room', 'Participants', 'Length_Hours', 'Contact_Hours']].head())
        
        # Render.com deployment
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        print(f"Traceback: {traceback.format_exc()}")