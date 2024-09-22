# Class for storing credentials
class Credentials:

    def __init__(self, status, usertype, userid, patient_id, doctor_id):
        # Constructor to initialize credentials
        self.status = status  # Status Success or Failed
        self.usertype = usertype  # Type of user
        self.userid = userid  # User ID
        self.patient_id = patient_id  # Patient ID (if applicable)
        self.doctor_id = doctor_id  # Doctor ID (if applicable)


# Class for storing patient details
class PatientDetails:

    def __init__(self, patient_id, patient_name, patient_surname, patient_gender, patient_age, patient_weight,
                 patient_diagnosis, patient_insul_per_carhyd, patient_blood_sugar_target, patient_correction_factor):
        # Constructor to initialize patient details
        self.patient_id = patient_id  # Patient ID
        self.patient_name = patient_name  # Patient's first name
        self.patient_surname = patient_surname  # Patient's last name
        self.patient_gender = patient_gender  # Patient's gender
        self.patient_age = patient_age  # Patient's age
        self.patient_weight = patient_weight  # Patient's weight
        self.patient_diagnosis = patient_diagnosis  # Patient's diagnosis
        self.patient_insul_per_carhyd = patient_insul_per_carhyd  # Patient's insulin per carbohydrate ratio
        self.patient_blood_sugar_target = patient_blood_sugar_target  # Patient's blood sugar target
        self.patient_correction_factor = patient_correction_factor  # Patient's correction factor


# Class for storing doctor details
class DoctorDetails:

    def __init__(self, doctor_id, doctor_name, doctor_surname, doctor_office_address, doctor_age, doctor_email):
        # Constructor to initialize doctor details
        self.doctor_id = doctor_id  # Doctor ID
        self.doctor_name = doctor_name  # Doctor's first name
        self.doctor_surname = doctor_surname  # Doctor's last name
        self.doctor_office_address = doctor_office_address  # Doctor's office address
        self.doctor_age = doctor_age  # Doctor's age
        self.doctor_email = doctor_email  # Doctor's email address
