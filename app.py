import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from google import genai
from google.genai import types

global chat_session
chat_session = None
load_dotenv()
try:
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)
    GEMINI_MODEL_NAME = 'gemini-2.5-flash'
    chat_session = gemini_client.chats.create(
        model=GEMINI_MODEL_NAME,
        config=genai.types.GenerateContentConfig(
            system_instruction=
            "You are an expert system in the field of loans and credit risk. Your primary task is to provide concise, accurate, and helpful answers in based on the conversation context."
            "If you receive a message starting with 'SYSTEM LOG:', read the information and store it in context, but **do not generate a response** to the user for that log message."
            "column SK_ID_CURR means ID of loan in our sample."
            "column YEARS_BIRTH (Age) means Client's age in years at the time of application."
            "column YEARS_EMPLOYED means How many years before the application the person started current employment."
            "column YEARS_REGISTRATION means How many years before the application did client change his registration."
            "column YEARS_ID_PUBLISH means How many years before the application did client change the identity document with which he applied for the loan."
            "column CNT_FAM_MEMBERS means How many family members does client have."
            "column OCCUPATION_TYPE means What kind of occupation does the client have."
            "column ORGANIZATION_TYPE means Type of organization where client works."
            "column REGION_POPULATION_RELATIV means Normalized population of region where client lives (higher number means the client lives in more populated region)."
            "column AMT_INCOME_TOTAL means Income of the client."
            "column AMT_CREDIT means Credit amount of the loan."
            "column AMT_ANNUITY means Loan annuity."
            "column AMT_GOODS_PRICE For consumer loans it is the price of the goods for which the loan is given."
            "column EXT_SOURCE_2 means Normalized score from external data source."
            "column EXT_SOURCE_3 means Normalized score from external data source."
            "column OWN_CAR_AGE means Age of client's car."
            "column YEARS_LAST_PHONE_CHANGE means How many years before application did client change phone."
            "column AMT_REQ_CREDIT_BUREAU_YEAR means Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)."
            "column WEEKDAY_APPR_PROCESS_START means On which day of the week did the client apply for previous application."
            "column HOUR_APPR_PROCESS_START means Approximately at what day hour did the client apply for the previous application."
            "column OBS_30_CNT_SOCIAL_CIRCL means How many observation of client's social surroundings with observable 30 DPD (days past due) default."
            "column OBS_60_CNT_SOCIAL_CIRCLE means How many observation of client's social surroundings with observable 60 DPD (days past due) default."
            "column CNT_PAYMENT means Term of previous credit at application of the previous application."
            "column SELLERPLACE_AREA means Selling area of seller place of the previous application."
            "column YEARS_DECISION means Relative to current application when was the decision about previous application made."
            "column YEARS_FIRST_DUE means Relative to application date of current application when was the first due supposed to be of the previous application."
            "column YEARS_LAST_DUE means Relative to application date of current application when was the last due date of the previous application."
            "column YEARS_LAST_DUE_1ST_VERSION means Relative to application date of current application when was the first due of the previous application."
            "column YEARS_FIRST_DRAWING means Relative to application date of current application when was the first disbursement of the previous application."
            "column YEARS_TERMINATION means Relative to application date of current application when was the expected termination of the previous application."
            "column HOUR_APPR_PROCESS_START means Approximately at what hour did the client apply for the loan."
            "column NFLAG_INSURED_ON_APPROVAL means Did the client requested insurance during the previous application."
            "column PRODUCT_COMBINATION means Detailed product combination of the previous application."
            "column NAME_GOODS_CATEGORY means What kind of goods did the client apply for in the previous application."
            "column WEEKDAY_APPR_PROCESS_START means On which day of the week did the client apply for the loan."
            "If the column has previous in its name, it refers to the previous loan application history of the client (aggregated)."
        )
    )
except Exception as e:
    gemini_client = None


app = Flask(__name__)

MODEL_PATH = 'LoanDefaulter_LightGBM.pkl'
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{MODEL_PATH}' not found. Please place the .pkl file in the root directory.")
except Exception as e:
    print(f"Error loading model: {e}")

def get_val(value, type_func, default=0):
    try:
        return type_func(value)
    except (ValueError, TypeError):
        return default

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    prob = None
    raw = None
    error_msg = None

    if request.method == 'POST':
        if not model:
            error_msg = "Model not loaded. Please check server logs."
        else:
            try:
                input_data = {
                    'ORGANIZATION_TYPE': request.form.get('ORGANIZATION_TYPE'),
                    'EXT_SOURCE_3': get_val(request.form.get('EXT_SOURCE_3'), float),
                    'EXT_SOURCE_2': get_val(request.form.get('EXT_SOURCE_2'), float),
                    'YEARS_ID_PUBLISH': get_val(request.form.get('YEARS_ID_PUBLISH'), float),
                    'YEARS_EMPLOYED': get_val(request.form.get('YEARS_EMPLOYED'), float),
                    'YEARS_REGISTRATION': get_val(request.form.get('YEARS_REGISTRATION'), float),
                    'YEARS_BIRTH': get_val(request.form.get('YEARS_BIRTH'), float),
                    'AMT_ANNUITY': get_val(request.form.get('AMT_ANNUITY'), float),
                    'SK_ID_CURR': get_val(request.form.get('SK_ID_CURR'), int),
                    'REGION_POPULATION_RELATIVE': get_val(request.form.get('REGION_POPULATION_RELATIVE'), float),
                    'YEARS_LAST_PHONE_CHANGE': get_val(request.form.get('YEARS_LAST_PHONE_CHANGE'), float),
                    'PREV_SELLERPLACE_AREA_MEAN': get_val(request.form.get('PREV_SELLERPLACE_AREA_MEAN'), float),
                    'PREV_YEARS_DECISION_MEAN': get_val(request.form.get('PREV_YEARS_DECISION_MEAN'), float),
                    'AMT_CREDIT': get_val(request.form.get('AMT_CREDIT'), float),
                    'PREV_HOUR_APPR_PROCESS_START_MEAN': get_val(request.form.get('PREV_HOUR_APPR_PROCESS_START_MEAN'), float),
                    'PREV_YEARS_FIRST_DUE_MEAN': get_val(request.form.get('PREV_YEARS_FIRST_DUE_MEAN'), float),
                    'PREV_CNT_PAYMENT_MAX': get_val(request.form.get('PREV_CNT_PAYMENT_MAX'), float),
                    'AMT_INCOME_TOTAL': get_val(request.form.get('AMT_INCOME_TOTAL'), float),
                    'AMT_GOODS_PRICE': get_val(request.form.get('AMT_GOODS_PRICE'), float),
                    'PREV_AMT_ANNUITY_MEAN': get_val(request.form.get('PREV_AMT_ANNUITY_MEAN'), float),
                    'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN': get_val(request.form.get('PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN'), float),
                    'PREV_AMT_ANNUITY_MEDIAN': get_val(request.form.get('PREV_AMT_ANNUITY_MEDIAN'), float),
                    'PREV_SK_ID_PREV_COUNT': get_val(request.form.get('PREV_SK_ID_PREV_COUNT'), float),
                    'PREV_YEARS_TERMINATION_MEAN': get_val(request.form.get('PREV_YEARS_TERMINATION_MEAN'), float),
                    'PREV_AMT_CREDIT_MEDIAN': get_val(request.form.get('PREV_AMT_CREDIT_MEDIAN'), float),
                    'AMT_REQ_CREDIT_BUREAU_YEAR': get_val(request.form.get('AMT_REQ_CREDIT_BUREAU_YEAR'), float),
                    'PREV_YEARS_LAST_DUE_MEAN': get_val(request.form.get('PREV_YEARS_LAST_DUE_MEAN'), float),
                    'OCCUPATION_TYPE': request.form.get('OCCUPATION_TYPE'),
                    'PREV_AMT_APPLICATION_MEDIAN': get_val(request.form.get('PREV_AMT_APPLICATION_MEDIAN'), float),
                    'PREV_PRODUCT_COMBINATION_<LAMBDA>': request.form.get('PREV_PRODUCT_COMBINATION_LAMBDA'),
                    'PREV_AMT_CREDIT_MEAN': get_val(request.form.get('PREV_AMT_CREDIT_MEAN'), float),
                    'CNT_FAM_MEMBERS': get_val(request.form.get('CNT_FAM_MEMBERS'), float),
                    'OBS_30_CNT_SOCIAL_CIRCLE': get_val(request.form.get('OBS_30_CNT_SOCIAL_CIRCLE'), float),
                    'OWN_CAR_AGE': get_val(request.form.get('OWN_CAR_AGE'), float),
                    'PREV_AMT_APPLICATION_MEAN': get_val(request.form.get('PREV_AMT_APPLICATION_MEAN'), float),
                    'PREV_AMT_GOODS_PRICE_MEDIAN': get_val(request.form.get('PREV_AMT_GOODS_PRICE_MEDIAN'), float),
                    'HOUR_APPR_PROCESS_START': get_val(request.form.get('HOUR_APPR_PROCESS_START'), int),
                    'PREV_AMT_GOODS_PRICE_MEAN': get_val(request.form.get('PREV_AMT_GOODS_PRICE_MEAN'), float),
                    'PREV_YEARS_FIRST_DRAWING_MEAN': get_val(request.form.get('PREV_YEARS_FIRST_DRAWING_MEAN'), float),
                    'PREV_SK_ID_CURR_FIRST': get_val(request.form.get('PREV_SK_ID_CURR_FIRST'), float),
                    'OBS_60_CNT_SOCIAL_CIRCLE': get_val(request.form.get('OBS_60_CNT_SOCIAL_CIRCLE'), float),
                    'PREV_NAME_GOODS_CATEGORY_<LAMBDA>': request.form.get('PREV_NAME_GOODS_CATEGORY_LAMBDA'),
                    'PREV_NFLAG_INSURED_ON_APPROVAL_MAX': get_val(request.form.get('PREV_NFLAG_INSURED_ON_APPROVAL_MAX'), float),
                    'WEEKDAY_APPR_PROCESS_START': request.form.get('WEEKDAY_APPR_PROCESS_START'),
                    'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>': request.form.get('PREV_WEEKDAY_APPR_PROCESS_START_LAMBDA')
                }

                df = pd.DataFrame([input_data])

                categorical_cols = [
                    'ORGANIZATION_TYPE', 
                    'OCCUPATION_TYPE', 
                    'PREV_PRODUCT_COMBINATION_<LAMBDA>',
                    'PREV_NAME_GOODS_CATEGORY_<LAMBDA>',
                    'WEEKDAY_APPR_PROCESS_START',
                    'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>'
                ]

                for col in categorical_cols:
                    df[col] = df[col].astype('category')

                raw_prediction = model.predict(df, raw_score=True)[0]
                
                probability = 1 / (1 + np.exp(-raw_prediction))
                
                prediction_class = 1 if probability > 0.5 else 0
                
                prediction_result = "High Risk (Defaulter)" if prediction_class == 1 else "Low Risk (Non-Defaulter)"
                prob = f"{probability:.2%}"
                raw = f"{raw_prediction:.4f}"
                
                if chat_session:
                    log_message = (
                        f"SYSTEM LOG: New user loan data received. "
                        f"Inputs: Loan Credit={input_data['AMT_CREDIT']}, "
                        f"Income={input_data['AMT_INCOME_TOTAL']}, "
                        f"Ext. Score 3={input_data['EXT_SOURCE_3']}, "
                        f"Ext. Score 2={input_data['EXT_SOURCE_2']}, "
                        f"Years Employed={input_data['YEARS_EMPLOYED']} years, "
                        f"Years ID Publish={input_data['YEARS_ID_PUBLISH']} years, "
                        f"Years Registration={input_data['YEARS_REGISTRATION']} years, "
                        f"Years Birth={input_data['YEARS_BIRTH']} years, "
                        f"Annuity={input_data['AMT_ANNUITY']}, "
                        f"Goods Price={input_data['AMT_GOODS_PRICE']}, "
                        f"Organization Type={input_data['ORGANIZATION_TYPE']}, "
                        f"Occupation Type={input_data['OCCUPATION_TYPE']}, "
                        f"Family Members={input_data['CNT_FAM_MEMBERS']}, "
                        f"Region Population Relative={input_data['REGION_POPULATION_RELATIVE']}, "
                        f"Own Car Age={input_data['OWN_CAR_AGE']}, "
                        f"ID={input_data['SK_ID_CURR']}, "
                        f"**RESULT:** {prediction_result} with {prob} probability of default. "
                        f"Please acknowledge this data but **do not respond** to the user at this time."
                    )
                    
                    try:
                        chat_session.send_message(log_message) 
                    except Exception as e:
                        pass

            except Exception as e:
                error_msg = str(e)

    lists = {
        'occupation': ['Laborers', 'Accountants', 'Managers', 'Sales staff', 'Drivers', 'Core staff', 'Medicine staff', 'High skill tech staff', 'Secretaries', 'Waiters/barmen staff', 'Cooking staff', 'Realty agents', 'Cleaning staff', 'Low-skill Laborers', 'Private service staff', 'Security staff', 'HR staff', 'IT staff'],
        'organization': ['Business Entity Type 3', 'Government', 'Other', 'Trade: type 7', 'Business Entity Type 2', 'Security Ministries', 'Self-employed', 'Construction', 'Transport: type 4', 'Trade: type 2', 'Housing', 'Industry: type 3', 'Military', 'Trade: type 3', 'Business Entity Type 1', 'Industry: type 2', 'School', 'Kindergarten', 'Industry: type 9', 'Medicine', 'Emergency', 'Industry: type 11', 'Police', 'Industry: type 5', 'Industry: type 10', 'Postal', 'Industry: type 4', 'Agriculture', 'Bank', 'Industry: type 12', 'University', 'Transport: type 2', 'Services', 'Transport: type 3', 'Industry: type 7', 'Restaurant', 'Telecom', 'Security', 'Mobile', 'Industry: type 1', 'Cleaning', 'Insurance', 'Electricity', 'Religion', 'Advertising', 'Trade: type 1', 'Legal Services', 'Realtor', 'Trade: type 6', 'Culture', 'Hotel', 'Transport: type 1', 'Industry: type 6', 'Industry: type 13', 'Industry: type 8', 'Trade: type 4', 'Trade: type 5'],
        'weekday': ['WEDNESDAY', 'MONDAY', 'SUNDAY', 'THURSDAY', 'SATURDAY', 'FRIDAY', 'TUESDAY'],
        'prev_prod': ['POS other with interest', 'POS mobile without interest', 'POS household with interest', 'POS industry without interest', 'Cash X-Sell: high', 'POS industry with interest', 'Cash', 'Cash Street: low', 'POS mobile with interest', 'Cash Street: high', 'Card X-Sell', 'Cash X-Sell: low', 'Card Street', 'Cash X-Sell: middle', 'POS household without interest', 'Cash Street: middle', 'POS others without interest'],
        'prev_goods': ['Vehicles', 'Mobile', 'Audio/Video', 'Furniture', 'XNA', 'Clothing and Accessories', 'Computers', 'Consumer Electronics', 'Auto Accessories', 'Medicine', 'Photo / Cinema Equipment', 'Office Appliances', 'Jewelry', 'Construction Materials', 'Gardening', 'Homewares', 'Medical Supplies', 'Tourism', 'Insurance', 'Sport and Leisure', 'Other', 'Additional Service', 'Fitness', 'Education', 'Direct Sales'],
        'prev_weekday': ['SATURDAY', 'FRIDAY', 'TUESDAY', 'MONDAY', 'SUNDAY', 'WEDNESDAY', 'THURSDAY']
    }

    return render_template('index.html', result=prediction_result, prob=prob, raw=raw, error=error_msg, lists=lists)


@app.route('/chat_api', methods=['POST'])
def chat_api_route():
    from flask import jsonify 

    if not chat_session:
        return jsonify({"reply": "Sorry, the chatbot is currently unavailable. (Session not initialized)"}), 503
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({"reply": "Please send a message."}), 400    
    try:
        response = chat_session.send_message(user_message)
        reply = response.text
    except Exception as e:
        reply = "An error occurred while connecting to the model. Please try again later."

    return jsonify({"reply": reply})
if __name__ == '__main__':
    app.run(debug=True)
