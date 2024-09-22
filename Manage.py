# Importing necessary classes from other modules
from Objects import *  # Importing classes for storing credentials, patient details, and doctor details
from Database import *  # Importing functions for database operations


# Function to attempt login with provided username and password
def login_attempt(username, password):
    # Check credentials against database
    cred_results = check_credentials_db(username, password)

    # If login attempt failed, return failed credentials
    if cred_results == "Failed":
        return Credentials("Failed", "NaN", "NaN", "NaN", "Nan")

    # If login attempt successful, return credentials object
    else:
        (cred_username, cred_password, cred_type, cred_user_id) = cred_results

        # Determine user type and associated IDs
        if cred_type == "DOCTOR":
            patient_id = "NaN"
            doctor_id = cred_user_id

        else:
            if cred_type == "PATIENT":
                patient_id = cred_user_id
                doctor_id = "NaN"

            else:
                patient_id = "NaN"
                doctor_id = "NaN"

        # Return credentials object
        return Credentials("Success", cred_type, cred_user_id, patient_id, doctor_id)


# Function to attempt sign up a new doctor
def doctor_sign_up(name, surname, address, email, age, password):
    # Check credentials against database
    sign_up_results = doctor_sign_up_db(name, surname, address, email, age, password)

    # If login attempt failed, return NaN
    if sign_up_results == "Failed":
        return {
            "userId": "NaN",
            "message": "NaN"
        }

    # If sign up attempt successful, return user id
    else:
        # Return user id
        return {
            "userId": sign_up_results,
            "message": "Your user id / username is: " + str(sign_up_results)
        }


# Function to attempt sign up a new patient
def patient_sign_up(name, surname, gender, age, weight, diagnosis, address, email, doctor_id, password):
    # Check credentials against database
    sign_up_results = patient_sign_up_db(name, surname, gender, age, weight, diagnosis, address, email, doctor_id,
                                         password)

    # If login attempt failed, return NaN
    if sign_up_results == "Failed":
        print('1')
        return {
            "userId": "NaN",
            "message": "NaN"
        }

    else:
        # If the doctor id does not exist, return the following message
        if sign_up_results == "The doctor id isn't correct":
            print('2')
            return {
                "userId": "NaN",
                "message": "The doctor id isn't correct"
            }

        else:
            # If sign up attempt successful, return user id
            return {
                "userId": sign_up_results,
                "message": "Your user id / username is: " + str(sign_up_results)
            }


# Function to retrieve patient details based on patient ID
def get_patient_details(patient_id):
    # Retrieve patient details from database
    patient_details = retrieve_patient_details_db(patient_id)

    # If retrieval failed, return failed patient details
    if patient_details == "Failed":
        return PatientDetails("NaN", "NaN", "NaN", "NaN", "Nan",
                              "NaN", "NaN", "NaN",
                              "NaN", "NaN")

    # If retrieval successful, return patient details object
    else:
        (patient_id, patient_name, patient_surname, patient_gender, patient_age, patient_weight, patient_diagnosis,
         patient_insul_per_carhyd, patient_blood_sugar_target, patient_correction_factor) = patient_details

        return PatientDetails(patient_id, patient_name, patient_surname, patient_gender, patient_age, patient_weight,
                              patient_diagnosis, patient_insul_per_carhyd, patient_blood_sugar_target,
                              patient_correction_factor)


# Function to retrieve patient history based on patient ID
def get_patient_history(patient_id):
    # Function to retrieve patient history based on patient ID
    patient_history = retrieve_patient_history_db(patient_id)

    # If retrieval failed, return default data structure
    if patient_history == "Failed":
        return {
                "asc_number": 0,
                "glucose_before": "NaN",
                "insulin_dosage": "NaN",
                "glucose_after": "NaN",
                "food_carbo": "NaN",
                "timestamp": "NaN"
            }

    # If retrieval successful, format data and return as JSON
    else:
        data = []
        j = 1
        print(j)
        for i in patient_history:
            print(i)
            data.append({
                "asc_number": j,
                "glucose_before": i[0],
                "insulin_dosage": i[1],
                "glucose_after": i[2],
                "food_carbo": i[3],
                "timestamp": i[4]
            })
            j = j + 1

        return data


# Function to return list of foods
def return_food_list():
    # Retrieve list of foods from database
    foods_list = retrieve_food_list_db()

    # If retrieval failed, return default data structure
    if foods_list == 'Failed':
        return {
            "food_name": "NaN"
        }

    # If retrieval successful, format data and return as JSON
    else:
        data = []
        print("return_food_list")

        for i in foods_list:
            data.append({
                "food_name": i[0]
            })

        return data


# Function to return measurement type information for a given food
def return_food_dosage(food_name):
    # Retrieve measurement type information from database for the given food
    food_dosage = return_food_dosage_db(food_name)
    print(food_dosage)

    # If retrieval failed, return default data structure
    if food_dosage == 'Failed':
        return {
            "food_dosage": "NaN"
        }

    # If retrieval successful, format data and return as JSON
    else:
        cups, units, types = food_dosage[0]
        print(cups)
        print(units)
        print(types)

        if cups is None:
            print("ok units")
            return {
                "food_dosage": types + " Size Units"
            }
        else:
            print("ok cups")
            return {
                "food_dosage": "Cups"
            }


# Function to calculate insulin dose based on food intake
def return_food_insulin_dose(patient_id, foods, quantities, insulin, indicator):
    # Calculate insulin dose based on patient ID, foods, and quantities
    food_info = calc_food_insulin_dose_db(patient_id, foods, quantities)
    print(food_info)

    # If calculation failed, return default data structure
    if food_info == 'Failed':
        return {
            "message": "NaN",
            "insulin_dose": "NaN",
            "food_carbo": "NaN"
        }

    # If calculation successful, format data and return as JSON
    else:
        food_insulin_dose, food_carbo = food_info

        if insulin is None:
            message = "The Recommended Insulin Dose is " + str(food_insulin_dose) + " units"
        else:
            if indicator == '1':
                message = "The Recommended Insulin Dose is " + str(food_insulin_dose + insulin) + " units"
            else:
                message = "The Additional Recommended Insulin Dose is " + str(food_insulin_dose) + " units"

        return {
            "message": message,
            "insulin_dose": food_insulin_dose,
            "food_carbo": food_carbo
        }


# Function to predict insulin dosage based on patient's glucose level
def insulin_dose_predict(patient_id, glucose_before, method, carbo_insulin):
    # Predict the Insulin Dosage
    insulin_predict = regression_db(patient_id, glucose_before, method)
    print(insulin_predict)

    # If prediction failed, return default data structure
    if insulin_predict == 'Failed':
        return {
            "result": "NaN",
            "insulin_predict": "NaN"
        }

    # If prediction successful, process the result and return
    else:

        if insulin_predict == "Your Glucose is within the normal range":
            # If glucose within normal range, set insulin dosage to 0.0
            insulin_dosage = 0.0
            return {
                "result": "Your Glucose is within the normal range",
                "insulin_predict": insulin_dosage
            }

        else:
            if insulin_predict == "Your Glucose is low, you need to eat":
                # If glucose low, recommend taking carbohydrates or orange juice
                insulin_dosage = 0.0
                return {
                    "result": "Your Glucose is low, you need to eat 15-20 gr of Carbohydrate",
                    "insulin_predict": insulin_dosage
                }

            else:
                if carbo_insulin is None:
                    message = "The Recommended Insulin Dose is " + str(insulin_predict) + " units"
                else:
                    message = "The Additional Recommended Insulin Dose is " + str(insulin_predict) + " units"

                return {
                    "result": message,
                    "insulin_predict": insulin_predict
                }


# Function to load all messages for a user
def load_all_messages(user_id):
    # Retrieve messages from database for the user
    messages = retrieve_messages_db(user_id)
    print(messages)

    # If retrieval failed, return default data structure
    if messages == "Failed":
        return {
                "otherUser": "NaN",
                "content": "NaN",
                "markedAsRead": "NaN",
                "messageId": "NaN",
                "senderReceiver": "NaN"
            }

    # If retrieval successful, return messages
    else:
        return messages


# Function to return list of receivers
def return_receivers_list():
    # Retrieve list of receivers from database
    receivers_list = retrieve_receivers_list_db()

    # If retrieval failed, return default data structure
    if receivers_list == 'Failed':
        return {
            "userId": "NaN",
            "name": "NaN",
            "userType": "NaN"
        }

    # If retrieval successful, return receivers list
    else:
        return receivers_list


# Function to get remaining insertions for a patient
def get_remain_insertions(patient_id):
    # Retrieve remaining insertions from database for the patient
    remain_insertions = retrieve_remain_insertions_db(patient_id)

    # If retrieval failed, return default data structure
    if remain_insertions == "There are no remaining Insertions":
        return {
            "asc_number": 0,
            "glucose_before": "NaN",
            "insulin_dosage": "NaN",
            "food_insulin": "NaN",
            "timestamp": "NaN"
        }

    # If retrieval successful, format data and return as JSON
    else:
        data = []
        j = 1
        print(j)
        for i in remain_insertions:
            print(i)
            data.append({
                "asc_number": j,
                "glucose_before": i[0],
                "insulin_dosage": i[1],
                "food_insulin": i[2],
                "timestamp": i[3]
            })
            j = j + 1

        return data


# Function to predict insulin dosage needed for exercise
def exercise_dose_predict(patient_id, glucose_before, method):
    # Predict insulin dosage using linear regression model
    insulin_predict = regression_db(patient_id, glucose_before, method)
    print(insulin_predict)

    # If prediction failed, return default data structure
    if insulin_predict == 'Failed':
        return {
            "insulin_predict": "NaN"
        }

    # If prediction successful, process the result and return
    else:

        if insulin_predict == "Your Glucose is within the normal range":
            # If glucose within normal range, set insulin dosage to 0.0
            insulin_dosage = 0.0
            return {
                "result": "Your Glucose is within the normal range",
                "insulin_predict": insulin_dosage,
                "status": "202"
            }

        else:
            if insulin_predict == "Your Glucose is low, you need to eat":
                # If glucose low, recommend taking carbohydrates or orange juice
                insulin_dosage = 0.0
                return {
                    "result": "Your Glucose is low, take a gel of carbohydrates or an orange juice",
                    "insulin_predict": insulin_dosage,
                    "status": "202"
                }

            else:
                # insulin_predict = insulin_predict.tolist()[0]
                # insulin_predict = 2.496582
                # For other cases, return the predicted insulin dosage
                return {
                    "result": "Your Preferable Insulin Dose is " + str(insulin_predict) + " mg",
                    "insulin_predict": insulin_predict,
                    "status": "200"
                }


# Function to return available dates for a patient's data
def return_available_dates(patient_id, date):
    # Retrieve available dates from database for the patient
    available_dates = retrieve_available_dates_db(patient_id, date)

    # If retrieval failed, return default data structure
    if available_dates == 'Failed':
        return {
            "date": "NaN"
        }

    # If retrieval successful, format data and return as JSON
    else:
        data = []
        print("return_available_dates")

        for i in available_dates:
            data.append({
                "date": i[0]
            })

        print("ok dates")
        return data


# Function to return a message about glucose management based on patient's glucose levels
def return_diagram_message(patient_id, date_from, date_to):
    # Retrieve glucose levels for the patient on the specified date
    glucose_levels = retrieve_diagram_message_db(patient_id, date_from, date_to)
    print(glucose_levels)

    # If retrieval failed, return default data structure
    if glucose_levels == 'Failed':
        return {
            "message": "NaN",
            "status": "NaN"
        }

    # If retrieval successful, calculate glucose management status and return a message
    else:
        # Calculate glucose management status based on glucose levels
        total_count = len(glucose_levels)  # Total number of glucose measurements
        # Count of glucose levels in target range (80-180)
        count_in_range = sum(1 for glucose in glucose_levels if 80 <= glucose <= 180)

        # Calculate percentage of glucose levels in target range
        if total_count > 0:
            # Percentage of glucose levels in target range
            percentage_in_range = (count_in_range / total_count) * 100
        else:
            percentage_in_range = 0.0  # If no measurements, set percentage to 0

        # Determine glucose management status based on percentage
        if percentage_in_range >= 67:  # If more than 67% of glucose levels are in target range
            return {
                "glucose_message": "Your Glucose was well managed during the day",
                "status": "good"
            }
        else:
            return {
                "glucose_message": "Your Glucose was not well managed during the day",
                "status": "bad"
            }


# Function to get details of a doctor
def get_doctor_details(doctor_id):
    # Retrieve details of the doctor from database
    doctor_details = retrieve_doctor_details_db(doctor_id)

    # If retrieval failed, return default data structure
    if doctor_details == "Failed":
        return DoctorDetails("NaN", "NaN", "NaN", "NaN",
                             "Nan", "NaN")

    # If retrieval successful, format data and return as DoctorDetails object
    else:
        (doctor_id, doctor_name, doctor_surname, doctor_office_address, doctor_age, doctor_email) = doctor_details

        return DoctorDetails(doctor_id, doctor_name, doctor_surname, doctor_office_address, doctor_age, doctor_email)


# Function to return list of patients associated with a doctor
def return_list_of_patients(doctor_id):
    # Retrieve list of patients associated with the doctor from database
    list_of_patients = return_list_of_patients_db(doctor_id)

    # If retrieval failed, return default data structure
    if list_of_patients == 'Failed':
        return {
            "patient_id": "NaN",
            "name": "NaN",
            "surname": "NaN",
            "gender": "NaN",
            "age": "NaN",
            "weight": "NaN",
            "address": "NaN",
            "diagnosis": "NaN",
            "death": "NaN",
            "email": "NaN",
            "isul_per_carhyd": "NaN"
        }

    # If retrieval successful, format data and return as JSON
    else:
        data = []
        print("return_list_of_patients")

        for i in list_of_patients:
            data.append({
                "patient_id": i[0],
                "name": i[1],
                "surname": i[2],
                "gender": i[3],
                "age": i[4],
                "weight": i[5],
                "address": i[6],
                "diagnosis": i[7],
                "death": i[8],
                "email": i[9],
                "insul_per_carhyd": i[10],
            })

        return data
