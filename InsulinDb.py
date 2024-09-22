# Import necessary modules and functions
from flask import Flask, request, jsonify, send_file
from Manage import *
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime, timedelta
from Database import configure_db  # Import the function to configure the DB

# Initialize Flask application
app = Flask('InsulinDb')

# Initialize CORS to allow requests from any origin
cors = CORS(app)
configure_db(app)

# Endpoint for user login
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        data = request.json

        # Extract username and password from request data
        username = data.get('username')
        password = data.get('password')

        # Attempt to login using provided credentials
        credentials = login_attempt(username, password)

        # Prepare response with user details
        result = {
            'status': credentials.status,
            'userType': credentials.usertype,
            'userId': credentials.userid,
            'patientId': credentials.patient_id,
            'doctorId': credentials.doctor_id
        }
        return jsonify(result), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for user sign up
@app.route('/sign_up', methods=['POST'])
def sign_up():
    if request.method == 'POST':
        data = request.json

        # Extract username and password from request data
        user_type = data.get('userType')

        if user_type == 'DOCTOR':
            name = data.get('name')
            surname = data.get('surname')
            address = data.get('address')
            email = data.get('email')
            age = data.get('age')
            password = data.get('password')

            # Attempt to sign up a doctor
            user_id = doctor_sign_up(name, surname, address, email, age, password)

        else:
            name = data.get('name')
            surname = data.get('surname')
            gender = data.get('gender')
            age = data.get('age')
            weight = data.get('weight')
            diagnosis = data.get('diagnosis')
            address = data.get('address')
            email = data.get('email')
            doctor_id = data.get('doctorId')
            password = data.get('password')

            # Attempt to sign up a patient
            user_id = patient_sign_up(name, surname, gender, age, weight, diagnosis, address, email, doctor_id,
                                      password)
            print(user_id)

        return jsonify(user_id), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving patient details
@app.route('/patient_details', methods=['POST'])
def patient_details():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract patient ID from request data
        patient_id = data.get('patientId')

        print(patient_id)
        print(2)

        # Retrieve patient details from database
        patient_det = get_patient_details(patient_id)

        # Prepare response with patient details
        result = {
            'patientId': patient_det.patient_id,
            'name': patient_det.patient_name,
            'surname': patient_det.patient_surname,
            'gender': patient_det.patient_gender,
            'age': patient_det.patient_age,
            'weight': patient_det.patient_weight,
            'diagnosis': patient_det.patient_diagnosis,
            "insul_per_carhyd": patient_det.patient_insul_per_carhyd,
            "blood_sugar_target": patient_det.patient_blood_sugar_target,
            "correction_factor": patient_det.patient_correction_factor
        }
        print(result)
        return jsonify(result), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving patient medical history
@app.route('/patient_history', methods=['POST'])
def patient_history():
    if request.method == 'POST':
        data = request.json

        # Extract patient ID from request data
        patient_id = data.get('patientId')
        print(patient_id)

        # Retrieve patient history from database
        patient_hist = get_patient_history(patient_id)
        print(patient_hist)

        # Return patient history as JSON response
        return jsonify(patient_hist), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for inserting patient measurement data for Insulin dosage
@app.route('/patient_measure_insert', methods=['PUT'])
def patient_measure_insert():
    if request.method == 'PUT':
        data = request.json

        # Extract patient's measurement data from request
        patient_id = data.get('patientId')
        glucose_before = data.get('glucoseBefore')
        insulin_dosage = data.get('insulinDosage')
        food_carbo = data.get('foodCarbo')
        glucose_after = data.get('glucoseAfter')
        timestamp = data.get('timestamp')

        print(patient_id)
        print(glucose_before)
        print(insulin_dosage)
        print(glucose_after)
        print(food_carbo)
        print(timestamp)

        if timestamp == 'NaN':
            # Insert measurement data into database
            if insert_patient_measure_db(patient_id, glucose_before, insulin_dosage, glucose_after, food_carbo):
                return jsonify({'message': 'Success Insertion of Measurement'}), 200
            else:
                return jsonify({'message': 'Failed Insertion of Measurement'}), 405

        else:
            # Update measurement data into database
            if update_patient_measure_db(patient_id, glucose_before, insulin_dosage, glucose_after, food_carbo,
                                         timestamp):
                return jsonify({'message': 'Success Update of Measurement'}), 200
            else:
                return jsonify({'message': 'Failed Update of Measurement'}), 405

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for deleting patient measurement data for Insulin dosage
@app.route('/patient_measure_delete', methods=['PUT'])
def patient_measure_delete():
    if request.method == 'PUT':
        data = request.json

        # Extract patient's measurement data from request
        patient_id = data.get('patientId')
        timestamp = data.get('timestamp')

        print(patient_id)
        print(timestamp)

        # Delete measurement data into database
        if delete_patient_measure_db(patient_id, timestamp):
            return jsonify({'message': 'Success Delete of Measurement'}), 200
        else:
            return jsonify({'message': 'Failed Delete of Measurement'}), 405

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving list of foods
@app.route('/foods_list', methods=['GET'])
def foods_list():
    if request.method == 'GET':
        print("foods_list")

        # Retrieve list of foods from database
        food_list = return_food_list()
        print("ok")

        # Return food list as JSON response
        return jsonify(food_list), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving food data (measurement on Cups or Grams)
@app.route('/food_dosage', methods=['POST'])
def foods_dosage():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract food name from request data
        food_name = data.get('food')

        # Retrieve information for the specific food from database
        food_dosage = return_food_dosage(food_name)

        print("ok")
        print(food_dosage)
        # Return insulin dosage as JSON response
        return jsonify(food_dosage), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for calculating insulin dosage based on food intake
@app.route('/insulin_food_calculator', methods=['POST'])
def insulin_food_calc():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract data from request
        patient_id = data.get('patientId')
        foods = data.get('foods')
        quantities = data.get('quantity')
        insulin = data.get('insulinDose')
        indicator = data.get('indicator')
        print(foods)
        print(quantities)

        # Calculate insulin dosage based on food intake
        prediction = return_food_insulin_dose(patient_id, foods, quantities, insulin, indicator)
        print(prediction)

        # Return insulin dosage as JSON response
        return jsonify(prediction), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for inserting data from an Excel file into the database
@app.route('/excel_insertion', methods=['POST'])
def excel_insertion():
    if request.method == 'POST':
        # Check if the input is valid
        if 'excelFile' not in request.files or 'patientId' not in request.form:
            return jsonify({'message': 'No file part'})

        patient_id = request.form['patientId']
        file = request.files['excelFile']

        if file.filename == '':
            return jsonify({'No selected file'})

        if file:
            # Read the Excel file
            data = pd.read_excel(file, header=None)
            print(data)

            # Insert data into database
            manage_excel_db(patient_id, data)

            return jsonify({'message': 'File uploaded successfully'})

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for predicting insulin dosage based on glucose levels
@app.route('/insulin_prediction', methods=['POST'])
def insulin_prediction():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract data from request
        patient_id = data.get('patientId')
        glucose_before = data.get('glucoseBefore')
        method = data.get('method')
        carbo_insulin = data.get('carboInsulin')

        print(patient_id)
        print(glucose_before)

        # Predict insulin dosage based on glucose levels
        prediction = insulin_dose_predict(patient_id, glucose_before, method, carbo_insulin)
        print(prediction)
        print("ok")

        print(prediction)

        # Return prediction as JSON response
        if prediction['insulin_predict'] == 0.0:
            print(prediction)
            return jsonify(prediction), 200
        else:
            print(prediction)
#           prediction['insulin_predict'] = prediction['insulin_predict'].tolist()
            return jsonify(prediction), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for loading messages for a user
@app.route('/load_messages', methods=['POST'])
def load_messages():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract user ID from request data
        user_id = data.get('userId')
        print(user_id)

        # Load all messages for the user
        all_messages = load_all_messages(user_id)
        print(all_messages)

        # Return messages as JSON response
        return jsonify(all_messages), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving a list of message receivers for each user
@app.route('/list_of_receivers', methods=['GET'])
def list_of_receivers():
    if request.method == 'GET':
        print("receivers")

        # Retrieve list of receivers from database
        receivers_list = return_receivers_list()
        print("ok")

        # Return list of receivers as JSON response
        return jsonify(receivers_list), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for updating the status of a message to "read"
@app.route('/read_messages', methods=['PUT'])
def read_messages():
    if request.method == 'PUT':
        data = request.json

        # Extract message ID from request data
        message_id = data.get('messageId')
        user_id = data.get('userId')


        # Update message status to "read" in the database
        if update_read_messages_db(message_id, user_id):
            return jsonify({'message': 'Succeed to change message status to Read'}), 200
        else:
            return jsonify({'message': 'Failed to change message status to Read'}), 405

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for sending a message
@app.route('/send_messages', methods=['PUT'])
def send_messages():
    if request.method == 'PUT':
        data = request.json
        print(data)

        # Extract data from request
        user_id = data.get('userId')
        send_to = data.get('sendTo')
        content = data.get('content')
        print(user_id)
        print(send_to)
        print(content)

        # Insert new message into the database
        if insert_new_messages_db(user_id, send_to, content):
            return jsonify({'message': 'Succeed to insert new message'}), 200
        else:
            return jsonify({'message': 'Failed to insert new message'}), 405

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving remaining insulin insertions for a patient
@app.route('/remaining_insertions', methods=['POST'])
def remaining_insertions():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract patient ID from request data
        patient_id = data.get('patientId')
        print(patient_id)

        # Retrieve remaining insulin insertions for the patient
        remain_insertions = get_remain_insertions(patient_id)
        print(remain_insertions)

        # Return remaining insertions as JSON response
        return jsonify(remain_insertions), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for predicting insulin dosage based on exercise
@app.route('/exercise_insulin_prediction', methods=['POST'])
def exercise_insulin_prediction():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract data from request
        patient_id = data.get('patientId')
        glucose_before = data.get('glucoseBefore')
        method = data.get('method')

        print(patient_id)
        print(glucose_before)

        # Predict insulin dosage based on exercise
        prediction = exercise_dose_predict(patient_id, glucose_before, method)
        print(prediction)
        print("ok")

        print(prediction)

        # Return prediction as JSON response
        if prediction['insulin_predict'] == 0.0:
            print(prediction)
            return jsonify(prediction), 200
        else:
            print(prediction)
            prediction['insulin_predict'] = prediction['insulin_predict'].tolist()
            return jsonify(prediction), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for loading available dates for a patient's data
@app.route('/load_dates', methods=['POST'])
def load_dates():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract patient ID from request data
        patient_id = data.get('patientId')
        date = data.get('date')

        print(patient_id)

        # Retrieve available dates for the patient's data
        list_of_dates = return_available_dates(patient_id, date)

        # Return available dates as JSON response
        return jsonify(list_of_dates), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for loading diagrams for a patient's data
@app.route('/load_diagrams', methods=['POST'])
def load_diagrams():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract data from request
        diagram_type = data.get('diagramType')
        patient_id = data.get('patientId')
        date_from = data.get('dateFrom')
        date_to = data.get('dateTo')

        print(patient_id, diagram_type, date_from, date_to)

        # If date_to is not provided, use current date
        if date_to == 'NaN':
            date_to = datetime.now().date()
            print(date_to)

        # If date_from is 'NaN', calculate the date 15 days before today
        if date_from == 'NaN':
            date_from = datetime.now().date() - timedelta(days=15)
            print(date_from)

        # Load diagram based on diagram type
        if diagram_type == 'TimeInRange':
            try:
                # Generate Time in Range diagram
                image_bytes = time_in_range_db(patient_id, date_from, date_to)
                if image_bytes is not None:
                    return send_file(BytesIO(image_bytes), mimetype='image/png')
                else:
                    return jsonify({'message': 'Failed to generate diagram'}), 500

            except Exception as e:
                print("Error occurred while loading diagram:", e)
                return jsonify({'message': 'Internal Server Error'}), 500

        else:
            try:
                # Generate Insulin-Glucose Relationship diagram
                image_bytes = plot_insulin_glucose_relationship_db(patient_id, date_from, date_to)
                if image_bytes is not None:
                    # Return diagram image as file response
                    return send_file(BytesIO(image_bytes), mimetype='image/png')
                else:
                    return jsonify({'message': 'Failed to generate diagram'}), 500

            except Exception as e:
                print("Error occurred while loading diagram:", e)
                return jsonify({'message': 'Internal Server Error'}), 500

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for loading message related to a diagram
@app.route('/load_diagram_message', methods=['POST'])
def load_diagram_message():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract data from request
        patient_id = data.get('patientId')
        date_from = data.get('dateFrom')
        date_to = data.get('dateTo')

        print(patient_id, date_from, date_to)

        # Retrieve message related to the diagram
        diagram_message = return_diagram_message(patient_id, date_from, date_to)
        print(diagram_message)

        # Return diagram message as JSON response
        return jsonify(diagram_message), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving doctor details
@app.route('/doctor_details', methods=['POST'])
def doctor_details():
    if request.method == 'POST':
        data = request.json
        print(data)

        # Extract doctor ID from request data
        doctor_id = data.get('doctorId')

        print(doctor_id)

        # Retrieve doctor details from database
        doctor_det = get_doctor_details(doctor_id)

        # Prepare response with doctor details
        result = {
            'doctorId': doctor_det.doctor_id,
            'name': doctor_det.doctor_name,
            'surname': doctor_det.doctor_surname,
            'officeAddress': doctor_det.doctor_office_address,
            'age': doctor_det.doctor_age,
            'email': doctor_det.doctor_email,
        }

        # Return doctor details as JSON response
        return jsonify(result), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for retrieving list of patients for a doctor
@app.route('/list_of_patients', methods=['POST'])
def list_of_patients():
    if request.method == 'POST':
        print("list_of_patients")

        data = request.json
        print(data)

        # Extract doctor ID from request data
        doctor_id = data.get('doctorId')

        # Retrieve list of patients for the doctor from database
        list_of_patient = return_list_of_patients(doctor_id)
        print("ok")
        print(list_of_patient)

        # Return list of patients as JSON response
        return jsonify(list_of_patient), 200

    return jsonify({'message': 'Method Not Allowed'}), 405


# Endpoint for updating patient's information
@app.route('/change_patients_info', methods=['PUT'])
def change_patients_info():
    if request.method == 'PUT':
        data = request.json

        # Extract data from request
        patient_id = data.get('patientId')
        age = data.get('age')
        weight = data.get('weight')
        diagnosis = data.get('diagnosis')
        death = data.get('death')
        insul_per_carhyd = data.get("carbo")
        blood_sugar_target = data.get("bloodSugarTarget")
        correction_factor = data.get("correctionFactor")

        print(patient_id)
        print(insul_per_carhyd)

        # Update patient's data in the database
        if update_patient_info_db(patient_id, age, weight, diagnosis, death, insul_per_carhyd, blood_sugar_target,
                                  correction_factor):
            return jsonify({'message': 'Success Update of Patient Information'}), 200
        else:
            return jsonify({'message': 'Failed Update of Patient Information'}), 405

    return jsonify({'message': 'Method Not Allowed'}), 405


# Run the Flask application
if __name__ == '__main__':
    app.run(host="35.160.120.126", port=int(os.environ.get("PORT", 5000)))
