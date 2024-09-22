import sqlite3
from InsulinDb import app
from flask import g
from flask_sqlalchemy import SQLAlchemy

app.config['DATABASE'] = 'DB_Insulin.db'


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        # Enable foreign key constraints
        db.execute("PRAGMA foreign_keys = ON")
    return db


def check_credentials_db(username, password):
    try:
        # Execute the query
        print(username, password)
        print(type(username), type(password))

        cursor = get_db().cursor()
        #       cursor.execute("SELECT USERNAME, PASSWORD, TYPE, USER_ID "
        #                      "FROM CREDENTIALS "
        #                      "WHERE USERNAME = ? "
        #                      "AND PASSWORD =  ? ", (username, password))

        cursor.execute("SELECT USERNAME, PASSWORD, TYPE, USER_ID "
                       "FROM CREDENTIALS ")
        # Fetch the results
        cred_results = cursor.fetchone()
        num_rows = cursor.rowcount

        #       if num_rows == 0:
        if cred_results is None:
            print("It's Empty")
            print(cred_results)
            return "Failed"

        #        else:
        #            if num_rows > 1:
        #                return "Failed"

        else:
            return cred_results

    except Exception as e:
        print("Error occurred while fetching credentials:", e)
        return "Failed"
    finally:
        if 'cursor' in locals():
            cursor.close()


def retrieve_patient_details_db(patient_id):
    # Execute the query
    cursor = get_db().cursor()
    cursor.execute("SELECT PATIENT_ID, NAME, SURNAME, AGE, WEIGHT, DIAGNOSIS, DEATH "
                   "FROM PATIENTS "
                   "WHERE PATIENT_ID = %s ", patient_id)

    # Fetch the results
    patient_details = cursor.fetchone()
    num_rows = cursor.rowcount

    if num_rows == 0:
        return "Failed"

    else:
        if num_rows > 1:
            return "Failed"

        else:
            return patient_details


def retrieve_patient_history_db(patient_id):
    # Execute the query
    cursor = get_db().cursor()
    cursor.execute("SELECT GLUCOSE_BEFORE, INSULIN_DOSAGE, GLUCOSE_AFTER, TIMESTAMP "
                   "FROM PATIENTS_HISTORY "
                   "WHERE PATIENT_ID = %s ", patient_id)

    # Fetch the results
    patient_history = cursor.fetchall()
    num_rows = cursor.rowcount

    if num_rows == 0:
        return "Failed"

    else:
        return patient_history
