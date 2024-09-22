# Importing necessary libraries
import sqlite3
import math
# from InsulinDb import app  # Importing Flask app instance
from flask import g  # Importing Flask's 'g' object for global variables
import pandas as pd  # Importing pandas library for data manipulation
import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import statsmodels.api as sn  # Importing statsmodels for statistical analysis
from sklearn.model_selection import train_test_split  # Importing train_test_split for data splitting
from sklearn.linear_model import LinearRegression  # Importing LinearRegression for linear regression modeling
# Importing r2_score, mae, mse for evaluation metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime  # Importing datetime for handling date and time
from io import BytesIO  # Importing BytesIO for handling binary data
from matplotlib.dates import DateFormatter  # Importing DateFormatter for date formatting
import matplotlib  # Importing matplotlib for plotting
# Importing RandomForestRegressor for Random Forest regression modeling
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress  # Import linregress for adding the Linear Regression line to the plot
from scipy.stats import t, ttest_rel
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')  # Using 'Agg' backend for matplotlib (no UI required)


# Setting the path to the SQLite database file
# app.config['DATABASE'] = 'DB_Insulin.db'

# Function to configure the database with the app instance
def configure_db():
    from InsulinDb import app
    app.config['DATABASE'] = 'DB_Insulin.db'


# Example database function
def get_db_connection():
    from InsulinDb import app
    conn = sqlite3.connect(g.app.config['DATABASE'])
    return conn
    
# Function to establish connection to the SQLite database
def get_db():
    from InsulinDb import app
    db = getattr(g, '_database', None)
    if db is None:
        # If no existing database connection, establish a new one
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        # Enable foreign key constraints
        db.execute("PRAGMA foreign_keys = ON")
    return db


# Function to check user credentials in the database
def check_credentials_db(username, password):
    try:
        # Execute the query to check credentials
        print(username, password)
        print(type(username), type(password))

        cursor = get_db().cursor()
        cursor.execute("SELECT USERNAME, PASSWORD, TYPE, USER_ID "
                       "FROM CREDENTIALS "
                       "WHERE USERNAME = ? "
                       "AND PASSWORD = ? ", (username, password))

        # Fetch the results
        cred_results = cursor.fetchone()

        if cred_results is None:
            # If no matching credentials found, return 'Failed'
            print("It's Empty")
            print(cred_results)
            return "Failed"

        else:
            # If credentials found, return the results
            print(cred_results)
            return cred_results

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching credentials:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to insert new doctor into the database
def doctor_sign_up_db(name, surname, address, email, age, password):
    try:
        # Execute the query to check credentials
        cursor = get_db().cursor()
        cursor.execute("SELECT MAX(USER_ID) "
                       "FROM CREDENTIALS")

        # Fetch the results
        cred_results = cursor.fetchone()

        if cred_results is None or cred_results[0] is None:
            # If no matching credentials found, return 'Failed'
            print("It's Empty")
            print(cred_results)
            return "Failed"

        else:
            # If credentials found, insert the new doctor into the database
            print(cred_results)

            # Assign value to the constants
            user_id = int(cred_results[0]) + 1
            ind_type = '2'
            user_type = 'DOCTOR'

            # Insert the new user into the table with the Credentials
            cursor.execute("INSERT INTO CREDENTIALS VALUES "
                           "(?, ?, ?, ?, ?) ",
                           (ind_type, user_type, user_id, password, user_id))

            print("insert cred")

            # Insert the new user into the table with the doctor's information
            cursor.execute("INSERT INTO DOCTORS VALUES "
                           "(?, ?, ?, ?, ?, ?) ",
                           (user_id, name, surname, address, age, email))

            # Commit the insertion
            get_db().commit()
            print("insert doctor")

            return user_id

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while inserting new doctor credentials:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to insert new doctor into the database
def patient_sign_up_db(name, surname, gender, age, weight, diagnosis, address, email, doctor_id, password):
    try:
        # Execute the query to check credentials
        cursor = get_db().cursor()
        cursor.execute("SELECT MAX(USER_ID) "
                       "FROM CREDENTIALS")

        # Fetch the results
        cred_results = cursor.fetchone()

        if cred_results is None or cred_results[0] is None:
            # If no matching credentials found, return 'Failed'
            print("It's Empty")
            print(cred_results)
            return "Failed"

        else:
            # Check if the doctor id the patient used, exists
            cursor.execute("SELECT DISTINCT DOCTOR_ID "
                           "FROM DOCTORS "
                           "WHERE DOCTOR_ID = ?", (doctor_id,))

            # Fetch the results
            doctor_exist = cursor.fetchone()

            if doctor_exist is None:
                # If no matching credentials found, return 'Failed'
                print("It's Empty")
                print(doctor_exist, 'doctor exist')
                return "The doctor id isn't correct"

            else:

                # If credentials found, insert the new patient into the database
                print(cred_results)

                # Assign value to the constants
                user_id = int(cred_results[0]) + 1
                print(user_id)
                ind_type = '1'
                user_type = 'PATIENT'
                death = 'NA'
                insul_per_carbo = 1.2
                correction_factor = 50
                blood_sugar_target = 95

                # Insert the new user into the table with the Credentials
                cursor.execute("INSERT INTO CREDENTIALS VALUES "
                               "(?, ?, ?, ?, ?) ",
                               (ind_type, user_type, user_id, password, user_id))

                print("insert cred")

                # Insert the new user into the table with the patient's information
                cursor.execute("INSERT INTO PATIENTS VALUES "
                               "(?, ?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?, ?, ?) ",
                               (user_id, doctor_id, name, surname, gender, age, weight, address, diagnosis,
                                death, email, insul_per_carbo, correction_factor, blood_sugar_target))

                # Commit the insertion
                get_db().commit()
                print("insert patient")

                return user_id

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while inserting new doctor credentials:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve patient details from the database
def retrieve_patient_details_db(patient_id):
    try:
        print(patient_id)
        print(type(patient_id))
        print(1)
        # Execute the query to retrieve patient details
        cursor = get_db().cursor()
            
        cursor.execute("SELECT PATIENT_ID, NAME, SURNAME, GENDER, AGE, WEIGHT, DIAGNOSIS, INSUL_PER_CARHYD,"
                       "BLOOD_SUGAR_TARGET, CORRECTION_FACTOR "
                       "FROM PATIENTS "
                       "WHERE PATIENT_ID = ? ", (patient_id,))

        # Fetch the results
        patient_details = cursor.fetchone()

        if patient_details is None:
            # If no patient details found, return 'Failed'
            print("It's Empty")
            print(patient_details)
            return "Failed"

        else:
            # If patient details found, return the results
            print(patient_details)
            return patient_details

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients details:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve patient's glucose history from the database
def retrieve_patient_history_db(patient_id):
    try:
        # Execute the query to retrieve patient's glucose history
        print(patient_id)
        cursor = get_db().cursor()
        cursor.execute("SELECT GLUCOSE_BEFORE, INSULIN_DOSE, GLUCOSE_AFTER, FOOD_CARBO, TIMESTAMP "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "ORDER BY TIMESTAMP DESC "
                       "LIMIT 100 ", (patient_id,))

        # Fetch the results
        patient_history = cursor.fetchall()
        print(patient_history)

        if patient_history is None:
            # If no patient history found, return 'Failed'
            print("failed")
            return "Failed"

        else:
            modified_patient_history = []
            for details in patient_history:
                details = list(details)  # Convert tuple to list
                glucose_after = details[2]

                if glucose_after == -1:
                    details[2] = None
                modified_patient_history.append(tuple(details))  # Convert back to tuple if needed

            # If patient history found, return the results
            print("ok patient history")
            return modified_patient_history

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients statistics:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to insert patient's measurements into the database
def insert_patient_measure_db(patient_id, glucose_before, insulin_dosage, glucose_after, food_carbo):
    try:
        # Execute the query to insert patient's measurements
        print(patient_id)

        cursor = get_db().cursor()
        cursor.execute("INSERT INTO PATIENTS_STATISTICS VALUES "
                       "(?, ?, ?, ?, ?, CURRENT_TIMESTAMP) ",
                       (patient_id, glucose_before, insulin_dosage, food_carbo, glucose_after))

        # Commit the insertion
        get_db().commit()
        return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients statistics:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to update the patient's history into the db
def update_patient_measure_db(patient_id, glucose_before_str, insulin_dosage_str, glucose_after_str, food_carbo_str,
                              timestamp):
    try:
        # Execute the query to update patient's measurements
        print(patient_id)
        print(timestamp)
        cursor = get_db().cursor()

        if glucose_after_str is None:
            glucose_after_star = '-1'

        try:
            # Convert the timestamp string to a datetime object
            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Handle the case where the timestamp  cannot be converted to datetime
            return "Failed"

        try:
            # Convert the glucose_after string to float
            glucose_after = float(glucose_after_str)
        except ValueError:
            # Handle the case where glucose_after_str is not a valid float
            return "Failed"

        try:
            # Convert the glucose_before string to float
            glucose_before = float(glucose_before_str)
        except ValueError:
            # Handle the case where glucose_before_str is not a valid float
            return "Failed"

        try:
            # Convert the insulin_dosage string to float
            insulin_dosage = float(insulin_dosage_str)
        except ValueError:
            # Handle the case where insulin_dosage_str is not a valid float
            return "Failed"

        try:
            # Convert the food_carbo string to float
            food_carbo = float(food_carbo_str)
        except ValueError:
            # Handle the case where food_carbo_str is not a valid float
            return "Failed"

        # Update the measurements in the database
        cursor.execute("UPDATE PATIENTS_STATISTICS "
                       "SET GLUCOSE_AFTER = ? "
                       ", GLUCOSE_BEFORE = ? "
                       ", INSULIN_DOSE = ? "
                       ", FOOD_CARBO = ? "
                       "WHERE TIMESTAMP = ? "
                       "AND PATIENT_ID = ?",
                       (glucose_after, glucose_before, insulin_dosage, food_carbo, timestamp, patient_id))

        # Commit the update
        get_db().commit()

        # Return success message
        return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while updating measurements for patient:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to delete the patient's history into the db
def delete_patient_measure_db(patient_id, timestamp):
    try:
        # Execute the query to delete patient's measurements
        print(patient_id)
        print(timestamp)
        cursor = get_db().cursor()

        # Delete the measurements in the database
        cursor.execute("DELETE FROM PATIENTS_STATISTICS "
                       "WHERE TIMESTAMP = ? "
                       "AND PATIENT_ID = ?",
                       (timestamp, patient_id))

        # Commit the delete
        get_db().commit()

        # Return success message
        return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while deleting measurements for patient:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve the list of foods from the database
def retrieve_food_list_db():
    try:
        # Execute the query to retrieve the list of foods
        cursor = get_db().cursor()
        cursor.execute("SELECT CARBOHYDRATE "
                       "FROM FOODS ")

        # Fetch the results
        foods_list = cursor.fetchall()
        print(foods_list)

        if foods_list is None:
            # If no food list found, return 'Failed'
            print("Failed")
            return "Failed"

        else:
            # If food list found, return the results
            print("ok food_list")
            return foods_list

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching food list:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve the food dosage from the database
def return_food_dosage_db(food_name):
    try:
        # Execute the query to fetch food dosage information
        print(food_name)
        cursor = get_db().cursor()
        cursor.execute("SELECT CUPS, UNITS, TYPE "
                       "FROM FOODS "
                       "WHERE CARBOHYDRATE = ? ", (food_name,))

        # Fetch all the results
        food_dosage = cursor.fetchall()
        print(food_dosage)

        if food_dosage is None:
            # If no food dosage found, return 'Failed'
            return "Failed"

        else:
            # If food dosage found, return the results
            print("ok food dosage")
            return food_dosage

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching food dosage:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to calculate the insulin dosage based on carbohydrate intake
def calc_food_insulin_dose_db(patient_id, foods, quantities):
    try:
        # Execute the query to fetch insulin per carbohydrate information
        cursor = get_db().cursor()
        cursor.execute("SELECT INSUL_PER_CARHYD "
                       "FROM PATIENTS "
                       "WHERE PATIENT_ID = ? ", (patient_id,))

        # Fetch the results
        insul_per_carhyd = cursor.fetchone()

        print("insul_per_carhyd", insul_per_carhyd)

        # Check if any data was fetched
        if insul_per_carhyd is None:
            return "Failed"

        else:
            # Initialize total carbohydrate intake
            carbohydrate = 0.0

            # Loop through the provided foods and quantities
            for food, quantity in zip(foods, quantities):

                # Execute a query to fetch information about the food from the database
                cursor.execute("SELECT CUPS, UNITS "
                               "FROM FOODS "
                               "WHERE CARBOHYDRATE = ? ", (food,))

            # Fetch the results
                food_data = cursor.fetchone()

                # Check if any data was fetched
                if food_data is None:
                    return "Failed"

                else:
                    # Unpack the fetched data
                    (cups, units) = food_data
                    print("cups", cups)
                    print("units", units)

                    # Calculate carbohydrate intake based on cups or units
                    if cups is None:
                        carbohydrate += float(quantity)/units
                        print("carbohydrate", carbohydrate)
                    else:
                        carbohydrate += float(quantity)/cups
                        print("carbohydrate", carbohydrate)

                    print("insul_per_carhyd", insul_per_carhyd)
                    print("carbohydrate", carbohydrate)

            # Calculate insulin dose based on carbohydrate intake and insulin per carbohydrate ratio
            insulin_dose = round(insul_per_carhyd[0] * carbohydrate)
            print("insulin_dose", insulin_dose)

            return insulin_dose, carbohydrate

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching insulin dosage:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to insert insulin - glucose measurements from an excel
def manage_excel_db(patient_id, file):
    try:
        # Connect to the database
        print(patient_id)
        print(file)
        cursor = get_db().cursor()

        # Iterate through rows in the provided Excel file
        for index, row in file.iterrows():
            glucose_before = row[0]
            insulin_dosage = row[1]
            glucose_after = row[2]
            food_carbo = row[3]
            timestamp = row[4].strftime('%Y-%m-%d %H:%M:%S')

            # Insert data into the PATIENTS_STATISTICS table from the excel
            cursor.execute("INSERT INTO PATIENTS_STATISTICS VALUES "
                           "(?, ?, ?, ?, ?, ?) ",
                           (patient_id, glucose_before, insulin_dosage, food_carbo, glucose_after,
                            timestamp))

            # Commit the insertion
            get_db().commit()

        return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching excel data:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to adjust insulin dose based on food_carbo
def adjust_insulin_db(insulin_dose, food_carbo, patient_id):
    try:
        # Check if food_carbo is not zero
        if food_carbo != 0:
            # Execute the query to fetch insulin per carbohydrate information
            cursor = get_db().cursor()
            cursor.execute("SELECT INSUL_PER_CARHYD "
                           "FROM PATIENTS "
                           "WHERE PATIENT_ID = ? ", (patient_id,))

            # Fetch the results
            insul_per_carhyd = cursor.fetchone()

            # Check if any data was fetched
            if insul_per_carhyd is None:
                return "Failed"
            else:
                # Adjust insulin dose based on insulin per carbohydrate and food_carbo
                insulin_dose = insulin_dose - round(insul_per_carhyd[0] * food_carbo)
                return insulin_dose

        else:
            return insulin_dose  # Otherwise, keep the insulin dose unchanged

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching insulin per carbohydrate:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to predict insulin intake based on patient's glucose, using Linear Regression or Random Forest Regression
def regression_db(patient_id, glucose_measurement, method):
    try:
        print(patient_id)
        print(glucose_measurement)

        zero_insulin = 0

        # Execute the query to retrieve patients' insulin information
        cursor = get_db().cursor()
        cursor.execute("SELECT GLUCOSE_BEFORE, INSULIN_DOSE, FOOD_CARBO "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "AND INSULIN_DOSE > ?", (patient_id, zero_insulin))

        # Fetch the results
        patient_statistics = cursor.fetchall()
        print(patient_statistics)

        # Print the number of fetched rows
        number_of_rows = len(patient_statistics)
        print(number_of_rows)

        # Execute the query to retrieve patients' information
        cursor = get_db().cursor()
        cursor.execute("SELECT CORRECTION_FACTOR, BLOOD_SUGAR_TARGET "
                       "FROM PATIENTS "
                       "WHERE PATIENT_ID = ?", (patient_id,))

        # Fetch the results
        patient_info = cursor.fetchone()
        print(patient_info)

        if not patient_info:
            return "Failed"

        else:
            # Ensure glucose_before is numeric
            glucose_measurement = int(glucose_measurement)
            correction_factor, blood_sugar_target = patient_info

            # Perform glucose level check
            if glucose_measurement < 80:
                return "Your Glucose is low, you need to eat"
            else:
                if glucose_measurement < blood_sugar_target:
                    return "Your Glucose is within the normal range"

            if number_of_rows < 8 or patient_statistics is None:
                # Calculate excess glucose
                excess_glucose = glucose_measurement - blood_sugar_target

                # Calculate the prediction
                prediction = excess_glucose / correction_factor

                # Apply ceiling to the prediction
                prediction_rounded = round(prediction)

                return prediction_rounded

            # Prepare a list to store updated patient statistics
            updated_statistics = []

            # Iterate through each row in patient_statistics
            for row in patient_statistics:
                glucose_before, insulin_dose, food_carbo = row

                # Check the value of food_carbo and adjust insulin dose accordingly
                adjusted_insulin_dose = adjust_insulin_db(insulin_dose, food_carbo, patient_id)

                if adjusted_insulin_dose == "Failed":
                    return "Failed"
                else:
                    # Create a new tuple with the adjusted insulin dose
                    updated_row = (glucose_before, adjusted_insulin_dose)
                    updated_statistics.append(updated_row)

            # Create a DataFrame from the updated statistics
            data = pd.DataFrame(updated_statistics, columns=['GLUCOSE_BEFORE', 'INSULIN_DOSE'])

            # Calculate EDD and save to file
            edd = data.describe()

            print(edd)

            edd.to_csv('edd.csv', index=False)

            # Calculate correlation and save to file
            correl = data.corr()
            print(correl)
            correl.to_csv('correl.csv', index=False)

            # Extract features (X) and target variable (y) from patient statistics
            x = data[['GLUCOSE_BEFORE']]  # Independent variable: Glucose measurement
            y = data['INSULIN_DOSE']  # Dependent variable: Insulin dose

            # Split data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
            print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

            confidence_level = 0.95
            degrees_freedom = len(y_train) - 1
            t_value = t.ppf(1 - (1 - confidence_level) / 2, degrees_freedom)

            if method == 'linear':
                # Perform linear regression
                lm = LinearRegression()
                lm.fit(x, y)
                print(lm.intercept_, lm.coef_)

                # Fit the model
                lm.fit(x_train, y_train)

                # Predictions
                y_test_pred = lm.predict(x_test)
                y_train_pred = lm.predict(x_train)

                # Calculate R-squared scores
                r2_test = r2_score(y_test, y_test_pred)
                print("R-squared score (test):", r2_test)
                r2_train = r2_score(y_train, y_train_pred)
                print("R-squared score (train):", r2_train)

                # Calculate mae, mse score
                mae = mean_absolute_error(y_train, y_train_pred)
                print("Mean Absolute Error:", mae)
                mse = mean_squared_error(y_train, y_train_pred)
                print("Mean Squared Error:", mse)

                # Calculate metrics
                lin_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                lin_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                lin_train_mae = mean_absolute_error(y_train, y_train_pred)
                lin_test_mae = mean_absolute_error(y_test, y_test_pred)

                # Paired t-test for linear regression
                lin_train_vs_test_p_lr = ttest_rel(y_train, y_train_pred).pvalue if len(y_train) > 1 else np.nan

                # Confidence intervals for Linear Regression
                confidence_level = 0.95
                degrees_freedom = len(y_train) - 1
                t_value = t.ppf(1 - (1 - confidence_level) / 2, degrees_freedom)
                mae_std = np.std([lin_train_mae, lin_test_mae])
                rmse_std = np.std([lin_train_rmse, lin_test_rmse])
                mae_ci = t_value * mae_std / np.sqrt(len(y_train))
                rmse_ci = t_value * rmse_std / np.sqrt(len(y_train))

                print(f"Linear Regression RMSE (train): {lin_train_rmse}, RMSE (test): {lin_test_rmse}")
                print(f"Linear Regression MAE (train): {lin_train_mae}, MAE (test): {lin_test_mae}")
                print(f"Linear Regression p-value: {lin_train_vs_test_p_lr}")
                print(f"Linear Regression MAE CI: [{lin_train_mae - mae_ci}, {lin_train_mae + mae_ci}]")
                print(f"Linear Regression RMSE CI: [{lin_train_rmse - rmse_ci}, {lin_train_rmse + rmse_ci}]")

                # Make prediction
                prediction = lm.predict([[glucose_measurement]])

            else:
                # Perform Random Forest regression
                rf = RandomForestRegressor(n_estimators=100, random_state=0)
                rf.fit(x_train, y_train)

                # Predictions
                y_train_pred = rf.predict(x_train)
                y_test_pred = rf.predict(x_test)

                # Calculate metrics
                rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                rf_train_mae = mean_absolute_error(y_train, y_train_pred)
                rf_test_mae = mean_absolute_error(y_test, y_test_pred)

                # Predict insulin dose for all glucose measurements
                y_pred = rf.predict(x)

                # Calculate R-squared, mae, mse score
                mae = mean_absolute_error(y, y_pred)
                print("Mean Absolute Error:", mae)
                mse = mean_squared_error(y, y_pred)
                print("Mean Squared Error:", mse)
                r_squared = r2_score(y, y_pred)
                print("R-squared score (test):", r_squared)

                # Paired t-test for Random Forest
                rf_train_vs_test_p_rf = ttest_rel(y_train, y_train_pred).pvalue if len(y_train) > 1 else np.nan

                # Confidence intervals for Random Forest
                mae_std = np.std([rf_train_mae, rf_test_mae])
                rmse_std = np.std([rf_train_rmse, rf_test_rmse])
                mae_ci = t_value * mae_std / np.sqrt(len(y_train))
                rmse_ci = t_value * rmse_std / np.sqrt(len(y_train))

                print(f"Random Forest RMSE (train): {rf_train_rmse}, RMSE (test): {rf_test_rmse}")
                print(f"Random Forest MAE (train): {rf_train_mae}, MAE (test): {rf_test_mae}")
                print(f"Random Forest p-value: {rf_train_vs_test_p_rf}")
                print(f"Random Forest MAE CI: [{rf_train_mae - mae_ci}, {rf_train_mae + mae_ci}]")
                print(f"Random Forest RMSE CI: [{rf_train_rmse - rmse_ci}, {rf_train_rmse + rmse_ci}]")

                # Make prediction using the trained model
                prediction = rf.predict([[glucose_measurement]])

            # Apply ceiling to the prediction
            prediction = round(prediction[0])

            print(prediction)
            return prediction

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while predicting insulin dose:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve messages per user_id
def retrieve_messages_db(user_id):
    try:
        # Execute SQL query to fetch messages for the user_id
        print(user_id)
        cursor = get_db().cursor()
        cursor.execute("SELECT SENDER, RECEIVER, MESSAGE, READ, TIMESTAMP, MESSAGE_ID "
                       "FROM MESSAGES "
                       "WHERE (RECEIVER = ? "
                       "OR SENDER = ?) "
                       "ORDER BY TIMESTAMP DESC ", (user_id, user_id))

        # Fetch all the results
        messages = cursor.fetchall()
        print(messages)

        if messages is None:
            # If no messages found, return a message indicating so
            print("failed")
            return "You have no Messages"

        else:
            # Initialize a list to store formatted message information
            messages_info = []

            # Iterate through each message and format its information
            for sender, receiver, message, read_message, timestamp, message_id in messages:

                # Separate if the user_id is the sender or the receiver of the message
                if user_id == sender:
                    other_user = receiver
                    user_type = 0
                else:
                    other_user = sender
                    user_type = 1

                # Execute a query to get the name of the sender (doctor)
                cursor.execute("SELECT NAME||' '||SURNAME "
                               "FROM DOCTORS "
                               "WHERE DOCTOR_ID = ?", (other_user,))

                # Fetch the results
                user_name = cursor.fetchone()

                if user_name is None:
                    # If sender is not a doctor, check if it's a patient
                    cursor.execute("SELECT NAME||' '||SURNAME "
                                   "FROM PATIENTS "
                                   "WHERE PATIENT_ID = ?", (other_user,))

                    user_name = cursor.fetchone()

                    if user_name is None:
                        # If neither a doctor nor a patient, return "Failed"
                        print("Failed")
                        return "Failed"

                user_name = user_name[0]

                # Check if the message is read or unread
                marked_as_read = "read" if read_message == 1 else "unread"

                # Append formatted message information to messages_info list
                messages_info.append({
                    "otherUser": user_name + " " + other_user,
                    "content": message,
                    "markedAsRead": marked_as_read,
                    "messageId": message_id,
                    "timestamp": timestamp,
                    "userType": user_type
                })

        print("ok messages info")
        return messages_info

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching messages:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve the list of all users eligible for communication via messages
def retrieve_receivers_list_db():
    try:
        # Execute SQL query to fetch user IDs and types (doctor or patient) excluding admins
        cursor = get_db().cursor()
        cursor.execute("SELECT USER_ID, TYPE "
                       "FROM CREDENTIALS "
                       "WHERE TYPE NOT IN ('ADMIN') "
                       "ORDER BY USER_ID")

        # Fetch all the results
        users = cursor.fetchall()
        print("ok users")

        # Check if any users were found
        if users is None:
            # If no users found, return "Failed"
            print("Failed")
            return "Failed"

        else:
            # Initialize a list to store formatted user information
            user_info = []

            # Iterate through each user and format its information
            for user_id, user_type in users:

                if user_type == "DOCTOR":
                    # If user is a doctor, fetch their name from the DOCTORS table
                    cursor.execute("SELECT DOCTOR_ID||' - '||NAME||' '||SURNAME "
                                   "FROM DOCTORS "
                                   "WHERE DOCTOR_ID = ?", (user_id,))

                    # Fetch the results
                    doctor_name = cursor.fetchone()

                    if doctor_name is None:
                        # If doctor name not found, return "Failed"
                        print("Failed")
                        return "Failed"
                    else:
                        # Format user information and append it to user_info list
                        doctor_name = doctor_name[0]

                        user_info.append({
                            'userId': user_id,
                            'userType': user_type,
                            'name': doctor_name
                        })

                else:
                    # If user is a patient, fetch their name from the PATIENTS table
                    cursor.execute("SELECT PATIENT_ID||' - '||NAME||' '||SURNAME "
                                   "FROM PATIENTS "
                                   "WHERE PATIENT_ID = ?", (user_id,))

                    # Fetch the results
                    patient_name = cursor.fetchone()

                    if patient_name is None:
                        # If patient name not found, return "Failed"
                        print("Failed")
                        return "Failed"
                    else:
                        # Format user information and append it to user_info list
                        patient_name = patient_name[0]

                        user_info.append({
                            'userId': user_id,
                            'userType': user_type,
                            'name': patient_name
                        })

            print("ok user info")
            # Return the formatted user information
            return user_info

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while retrieving receivers:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to update the read status of messages in the database
def update_read_messages_db(message_id, user_id):
    try:
        # Set the read status to 1 (read)
        read_message = 1
        print(user_id)

        # Update the read status of the message with the given message ID
        cursor = get_db().cursor()
        cursor.execute("UPDATE MESSAGES "
                       "SET READ = ? "
                       "WHERE MESSAGE_ID = ? "
                       "AND RECEIVER = ?", (read_message, message_id, user_id))

        # Commit the insertion
        get_db().commit()
        # Return success message
        return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while updating message status:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to insert new sent messages into the db
def insert_new_messages_db(user_id, send_to, content):
    try:
        # Execute the query
        print(user_id)

        # Fetch the maximum message ID
        cursor = get_db().cursor()
        cursor.execute("SELECT MAX(MESSAGE_ID) "
                       "FROM MESSAGES ")

        max_message_id = cursor.fetchone()[0]

        # If no message ID found, return "Failed"
        if max_message_id is None:
            return "Failed"

        else:
            # Increment the maximum message ID
            max_message_id += 1
            # Set the message as unread (0)
            unread_message = 0

            # Insert the new message into the database

            cursor.execute("INSERT INTO MESSAGES VALUES "
                           "(?, ?, ?, ?, CURRENT_TIMESTAMP, ?) ",
                           (user_id, send_to, content, unread_message, max_message_id))

            # Commit the insertion
            get_db().commit()

            return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while inserting new message:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve the first 20 records where the glucose_after information isn't inserted
def retrieve_remain_insertions_db(patient_id):
    try:
        # Execute the query
        print(patient_id)

        # Define a placeholder value for glucose_after
        glucose_after = -1

        # Retrieve the glucose_before, insulin_dose, food_insulin, and timestamp for the patient
        cursor = get_db().cursor()
        cursor.execute("SELECT GLUCOSE_BEFORE, INSULIN_DOSE, FOOD_INSULIN, TIMESTAMP "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "AND GLUCOSE_AFTER = ?"
                       "ORDER BY TIMESTAMP ASC "
                       "LIMIT 20 ", (patient_id, glucose_after))

        # Fetch the results
        remaining_insertions = cursor.fetchall()
        print(remaining_insertions)

        # If no remaining insertions found, return a failure message
        if remaining_insertions is None:
            print("failed")
            return "There are no remaining Insertions"

        else:
            # Return the remaining insertions
            print("ok patient history")
            return remaining_insertions

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients statistics:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve the dates that the patient has insertions
def retrieve_available_dates_db(patient_id, date):
    try:
        # Execute SQL query to select distinct dates for the patient
        cursor = get_db().cursor()
        cursor.execute("SELECT DISTINCT DATE(TIMESTAMP) "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "AND DATE(TIMESTAMP) >= ? "
                       "ORDER BY DATE(TIMESTAMP) DESC", (patient_id, date))

        # Fetch the results
        available_dates = cursor.fetchall()
        print(available_dates)

        if available_dates is None:
            # If no available dates found, return a failure message
            print("Failed")
            return "Failed"

        else:
            print("ok available_dates")
            # Return the available dates
            return available_dates

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching available dates:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve patient's statistics in order to be used in the time in range diagram
def get_patient_statistics_db(patient_id, date_from, date_to):
    try:
        # Execute SQL query to select glucose levels and timestamps for the patient on the given date
        cursor = get_db().cursor()

        # Convert dates to strings in 'YYYY-MM-DD' format
        # date_from_str = date_from.strftime('%Y-%m-%d')
        # date_to_str = date_to.strftime('%Y-%m-%d')

        cursor.execute("SELECT GLUCOSE_BEFORE, TIMESTAMP "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "AND DATE(TIMESTAMP) BETWEEN ? AND ?", (patient_id, date_from, date_to))

        diagram_details = cursor.fetchall()

        if diagram_details is None:
            # If no statistics found, return None
            print("It's Empty Diagram")
            print(diagram_details)
            return None

        else:
            # Extract glucose levels and timestamps from the fetched data
            glucose_levels = [row[0] for row in diagram_details]
            timestamps = [row[1] for row in diagram_details]
            # Return the glucose levels and timestamps
            return glucose_levels, timestamps

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients statistics:", e)
        return None
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to create a time in range diagram
def time_in_range_db(patient_id, date_from, date_to):
    try:
        # Get glucose levels and timestamps for the patient on the given date
        glucose_levels, timestamps = get_patient_statistics_db(patient_id, date_from, date_to)

        # If data not available, return None
        if not glucose_levels or not timestamps:
            print("Failed to fetch patient statistics")
            return None

        # Convert timestamps to datetime objects
        timestamps = pd.to_datetime(timestamps)

        # Ensure data is sorted by timestamps
        data = sorted(zip(timestamps, glucose_levels), key=lambda x: x[0])
        timestamps, glucose_levels = zip(*data)

        # Define fixed color bands for glucose level ranges
        colors = ['yellow', 'green', 'red']
        color_limits = [80, 180, 500]  # Glucose level limits for color bands

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot colored background bands based on glucose level ranges
        for i in range(len(colors)):
            lower_limit = 0 if i == 0 else color_limits[i - 1]
            upper_limit = color_limits[i]
            ax.axhspan(lower_limit, upper_limit, facecolor=colors[i], alpha=0.3)

        # Plot glucose levels as a line graph
        ax.plot(timestamps, glucose_levels, marker='o', linestyle='-', color='black')

        # Calculate percentage of time in each glucose level range
        total_hours = len(timestamps)
        percentages = [(sum(lower_limit <= glucose < upper_limit for glucose in glucose_levels) / total_hours) * 100
                       for lower_limit, upper_limit in zip([0] + color_limits[:-1], color_limits)]

        # Create custom legends with percentage labels for background colors
        legend_labels = [f'Under 80 mg/dL ({percentages[0]:.0f}%)',
                         f'80-180 mg/dL ({percentages[1]:.0f}%)',
                         f'Above 180 mg/dL ({percentages[2]:.0f}%)']
        legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.3)
                          for i in range(len(colors))]
        ax.legend(legend_handles, legend_labels, loc='upper right')

        # Customize plot appearance
        ax.set_xlabel('Time')
        ax.set_ylabel('Glucose Level (mg/dL)')
        ax.set_title('Glucose Levels Throughout the Day')
        ax.grid(True)  # Add grid lines for better readability
        ax.set_ylim(0, 500)  # Adjust y-axis limits based on your data range

        # Format x-axis labels as hh:mm
        date_formatter = DateFormatter('%Y-%m-%d\n%H:%M')  # Define date formatter for yyyy-mm-dd hh:mm format
        ax.xaxis.set_major_formatter(date_formatter)  # Apply date formatter to x-axis labels
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Optionally adjust the number of ticks
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

        # Save the plot as a PNG image in memory
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        plt.close()  # Close the figure to free up resources
        image_bytes = image_stream.getvalue()  # Get the image bytes

        # Return the image as a response
        return image_bytes

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while creating time in range diagram:", e)
        return None


# Function to create a message based on the time in range diagram
def retrieve_diagram_message_db(patient_id, date_from, date_to):
    try:
        # Execute SQL query to retrieve glucose levels for the patient on the given date
        cursor = get_db().cursor()
        cursor.execute("SELECT GLUCOSE_BEFORE "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "AND DATE(TIMESTAMP) BETWEEN ? AND ? ", (patient_id, date_from, date_to))

        diagram_details = cursor.fetchall()

        if diagram_details is None:
            # If no data found, return failure message
            print("It's Empty")
            print(diagram_details)
            return "Failed"

        else:
            # Extract glucose levels from the loaded data
            glucose_levels = [row[0] for row in diagram_details]

            # Return the glucose levels
            return glucose_levels

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients glucose:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve insulin and glucose information in order to be used in the insulin-glucose diagram
def get_insulin_glucose_data_db(patient_id, date_from, date_to):
    try:
        # Execute SQL query to retrieve glucose and insulin levels for the patient on the given date
        cursor = get_db().cursor()
        cursor.execute("SELECT GLUCOSE_BEFORE, INSULIN_DOSE, FOOD_CARBO "
                       "FROM PATIENTS_STATISTICS "
                       "WHERE PATIENT_ID = ? "
                       "AND DATE(TIMESTAMP) >= ? "
                       "AND DATE(TIMESTAMP) <= ?", (patient_id, date_from, date_to))

        diagram_details = cursor.fetchall()

        if diagram_details is None:
            # If no data found, return None
            print("It's Empty")
            print(diagram_details)
            return None

        else:
            # Prepare a list to store updated patient details
            updated_details = []

            # Iterate through each row in diagram_details
            for row in diagram_details:
                glucose_before, insulin_dose, food_carbo = row

                # Check the value of food_carbo and adjust insulin dose accordingly
                adjusted_insulin_dose = adjust_insulin_db(insulin_dose, food_carbo, patient_id)

                if adjusted_insulin_dose == "Failed":
                    return "Failed"
                else:
                    # Create a new tuple with the adjusted insulin dose
                    updated_row = (glucose_before, adjusted_insulin_dose)
                    updated_details.append(updated_row)

            # Extract glucose and insulin levels from the loaded data
            glucose_levels = [row[0] for row in updated_details]
            insulin_levels = [row[1] for row in updated_details]

            # Return the glucose and insulin levels
            return glucose_levels, insulin_levels

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching patients statistics:", e)
        return None
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to create an insulin - glucose diagram
def plot_insulin_glucose_relationship_db(patient_id, date_from, date_to):
    try:
        # Get glucose and insulin levels for the patient on the given date
        glucose_levels, insulin_levels = get_insulin_glucose_data_db(patient_id, date_from, date_to)

        if not glucose_levels or not insulin_levels:
            # If data not available, return None
            print("Failed to fetch patient statistics")
            return None

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(glucose_levels, insulin_levels, color='blue', alpha=0.5)

        # Calculate linear regression parameters
        slope, intercept, r_value, p_value, std_err = linregress(glucose_levels, insulin_levels)

        # Add the regression line to the plot
        regression_line = [slope * x + intercept for x in glucose_levels]
        plt.plot(glucose_levels, regression_line, color='red', linestyle='--', linewidth=2)

        # Add labels and title
        plt.ylabel('Insulin Level (units)')
        plt.xlabel('Glucose Level (mg/dL)')
        plt.title('Plot Between Insulin and Glucose Levels')

        # Show grid
        plt.grid(True)

        # Save the plot as a PNG image in memory
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        plt.close()  # Close the figure to free up resources
        image_bytes = image_stream.getvalue()  # Get the image bytes

        # Return the image as a response
        return image_bytes

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while creating insulin glucose diagram:", e)
        return None


# Function to retrieve list of patients per doctor
def return_list_of_patients_db(doctor_id):
    try:
        # Execute SQL query to retrieve patient details for the given doctor
        cursor = get_db().cursor()
        cursor.execute("SELECT PATIENT_ID, NAME, SURNAME, GENDER, AGE, WEIGHT, ADDRESS, DIAGNOSIS, DEATH, EMAIL, "
                       "INSUL_PER_CARHYD FROM PATIENTS "
                       "WHERE DOCTOR_ID = ?", (doctor_id,))

        # Fetch the results
        list_of_patients = cursor.fetchall()
        print(list_of_patients)

        if list_of_patients is None:
            # If no data found, return failure message
            print("Failed")
            return "Failed"

        else:
            # Return the list of patients
            print("ok list_of_patients")
            return list_of_patients

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching list of patients:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to retrieve doctor details
def retrieve_doctor_details_db(doctor_id):
    try:
        print(doctor_id)

        # Execute SQL query to retrieve doctor details for the given doctor ID
        cursor = get_db().cursor()
        cursor.execute("SELECT DOCTOR_ID, NAME, SURNAME, OFFICE_ADDRESS, AGE, EMAIL "
                       "FROM DOCTORS "
                       "WHERE DOCTOR_ID = ? ", (doctor_id,))

        # Fetch the results
        doctor_details = cursor.fetchone()

        if doctor_details is None:
            # If no data found, return failure message
            print("It's Empty")
            print(doctor_details)
            return "Failed"

        else:
            # Return doctor's details
            print(doctor_details)
            return doctor_details

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while fetching doctors details:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()


# Function to update patient's information
def update_patient_info_db(patient_id, age, weight, diagnosis, death, insul_per_carhyd, blood_sugar_target,
                           correction_factor):
    try:
        # Execute the query to update patient's information
        print(patient_id, insul_per_carhyd)

        cursor = get_db().cursor()
        cursor.execute("UPDATE PATIENTS "
                       "SET INSUL_PER_CARHYD = ? "
                       ",BLOOD_SUGAR_TARGET = ? "
                       ",CORRECTION_FACTOR = ? "
                       ",AGE = ?"
                       ",WEIGHT = ?"
                       ",DIAGNOSIS = ?"
                       ",DEATH = ?"
                       "WHERE PATIENT_ID = ?", (insul_per_carhyd, blood_sugar_target, correction_factor, age,
                                                weight, diagnosis, death, patient_id))

        # Commit the insertion
        get_db().commit()
        return "Success"

    except Exception as e:
        # Print an error message if an exception occurs
        print("Error occurred while updating patient's details:", e)
        return "Failed"
    finally:
        # Close the cursor in the finally block to ensure resource cleanup
        if 'cursor' in locals():
            cursor.close()
