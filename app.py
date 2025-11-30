from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

class LoanDefaultPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load('LoanDefaulter_LightGBM.pkl')
            
            if hasattr(self.model, 'feature_name_'):
                self.features = self.model.feature_name_
            else:
                self.features = [
                    'ORGANIZATION_TYPE', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'YEARS_ID_PUBLISH', 
                    'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'YEARS_BIRTH', 'AMT_ANNUITY', 
                    'SK_ID_CURR', 'REGION_POPULATION_RELATIVE', 'YEARS_LAST_PHONE_CHANGE', 
                    'PREV_SELLERPLACE_AREA_MEAN', 'PREV_YEARS_DECISION_MEAN', 'AMT_CREDIT', 
                    'PREV_HOUR_APPR_PROCESS_START_MEAN', 'PREV_YEARS_FIRST_DUE_MEAN', 
                    'PREV_CNT_PAYMENT_MAX', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 
                    'PREV_AMT_ANNUITY_MEAN', 'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN', 
                    'PREV_AMT_ANNUITY_MEDIAN', 'PREV_SK_ID_PREV_COUNT', 
                    'PREV_YEARS_TERMINATION_MEAN', 'PREV_AMT_CREDIT_MEDIAN', 
                    'AMT_REQ_CREDIT_BUREAU_YEAR', 'PREV_YEARS_LAST_DUE_MEAN', 
                    'OCCUPATION_TYPE', 'PREV_AMT_APPLICATION_MEDIAN', 
                    'PREV_PRODUCT_COMBINATION_<LAMBDA>', 'PREV_AMT_CREDIT_MEAN', 
                    'CNT_FAM_MEMBERS', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OWN_CAR_AGE', 
                    'PREV_AMT_APPLICATION_MEAN', 'PREV_AMT_GOODS_PRICE_MEDIAN', 
                    'HOUR_APPR_PROCESS_START', 'PREV_AMT_GOODS_PRICE_MEAN', 
                    'PREV_YEARS_FIRST_DRAWING_MEAN', 'PREV_SK_ID_CURR_FIRST', 
                    'OBS_60_CNT_SOCIAL_CIRCLE', 'PREV_NAME_GOODS_CATEGORY_<LAMBDA>', 
                    'PREV_NFLAG_INSURED_ON_APPROVAL_MAX', 'WEEKDAY_APPR_PROCESS_START', 
                    'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>'
                ]
        except FileNotFoundError:
            print("❌ Model file 'LoanDefaulter_LightGBM.pkl' not found.")
            self.model = None
    
    def convert_to_model_format(self, inputs):
        model_inputs = inputs.copy()
        time_features = [
            'YEARS_BIRTH', 'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 
            'YEARS_ID_PUBLISH', 'YEARS_LAST_PHONE_CHANGE'
        ]
        for feat in time_features:
            if feat in model_inputs:
                model_inputs[feat] = -abs(float(model_inputs[feat]))
        return model_inputs
    
    def predict(self, input_data):
        if self.model is None:
            return 0.5, False, "Model not loaded"

        try:
            ordered_data = [input_data.get(f, 0) for f in self.features]
            input_df = pd.DataFrame([ordered_data], columns=self.features)

            input_df = input_df.infer_objects() 
            
            string_columns = input_df.select_dtypes(include=['object']).columns

            def encode_as_int(x):
                return abs(hash(str(x))) % 10_000_000  

            for col in string_columns:
                input_df[col] = input_df[col].apply(encode_as_int)

            X = input_df.to_numpy(dtype=float)

            default_prob = self.model.predict(X)[0]
            return default_prob, default_prob > 0.5, None

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return 0.5, False, str(e)

predictor = LoanDefaultPredictor()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        try:
            user_data = {
                'ORGANIZATION_TYPE': request.form.get('ORGANIZATION_TYPE'),
                'EXT_SOURCE_3': float(request.form.get('EXT_SOURCE_3')),
                'EXT_SOURCE_2': float(request.form.get('EXT_SOURCE_2')),
                'YEARS_ID_PUBLISH': float(request.form.get('YEARS_ID_PUBLISH')), 
                'YEARS_EMPLOYED': float(request.form.get('YEARS_EMPLOYED')),     
                'YEARS_REGISTRATION': float(request.form.get('YEARS_REGISTRATION')), 
                'YEARS_BIRTH': float(request.form.get('YEARS_BIRTH')),         
                'AMT_ANNUITY': float(request.form.get('AMT_ANNUITY')),
                'SK_ID_CURR': float(request.form.get('SK_ID_CURR')),
                'REGION_POPULATION_RELATIVE': float(request.form.get('REGION_POPULATION_RELATIVE')),
                'YEARS_LAST_PHONE_CHANGE': float(request.form.get('YEARS_LAST_PHONE_CHANGE')), 
                'PREV_SELLERPLACE_AREA_MEAN': float(request.form.get('PREV_SELLERPLACE_AREA_MEAN')),
                'PREV_YEARS_DECISION_MEAN': float(request.form.get('PREV_YEARS_DECISION_MEAN')),
                'AMT_CREDIT': float(request.form.get('AMT_CREDIT')),
                'PREV_HOUR_APPR_PROCESS_START_MEAN': float(request.form.get('PREV_HOUR_APPR_PROCESS_START_MEAN')),
                'PREV_YEARS_FIRST_DUE_MEAN': float(request.form.get('PREV_YEARS_FIRST_DUE_MEAN')),
                'PREV_CNT_PAYMENT_MAX': float(request.form.get('PREV_CNT_PAYMENT_MAX')),
                'AMT_INCOME_TOTAL': float(request.form.get('AMT_INCOME_TOTAL')),
                'AMT_GOODS_PRICE': float(request.form.get('AMT_GOODS_PRICE')),
                'PREV_AMT_ANNUITY_MEAN': float(request.form.get('PREV_AMT_ANNUITY_MEAN')),
                'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN': float(request.form.get('PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN')),
                'PREV_AMT_ANNUITY_MEDIAN': float(request.form.get('PREV_AMT_ANNUITY_MEDIAN')),
                'PREV_SK_ID_PREV_COUNT': float(request.form.get('PREV_SK_ID_PREV_COUNT')),
                'PREV_YEARS_TERMINATION_MEAN': float(request.form.get('PREV_YEARS_TERMINATION_MEAN')),
                'PREV_AMT_CREDIT_MEDIAN': float(request.form.get('PREV_AMT_CREDIT_MEDIAN')),
                'AMT_REQ_CREDIT_BUREAU_YEAR': float(request.form.get('AMT_REQ_CREDIT_BUREAU_YEAR')),
                'PREV_YEARS_LAST_DUE_MEAN': float(request.form.get('PREV_YEARS_LAST_DUE_MEAN')),
                'OCCUPATION_TYPE': request.form.get('OCCUPATION_TYPE'),
                'PREV_AMT_APPLICATION_MEDIAN': float(request.form.get('PREV_AMT_APPLICATION_MEDIAN')),
                'PREV_PRODUCT_COMBINATION_<LAMBDA>': float(request.form.get('PREV_PRODUCT_COMBINATION_<LAMBDA>')),
                'PREV_AMT_CREDIT_MEAN': float(request.form.get('PREV_AMT_CREDIT_MEAN')),
                'CNT_FAM_MEMBERS': float(request.form.get('CNT_FAM_MEMBERS')),
                'OBS_30_CNT_SOCIAL_CIRCLE': float(request.form.get('OBS_30_CNT_SOCIAL_CIRCLE')),
                'OWN_CAR_AGE': float(request.form.get('OWN_CAR_AGE')),
                'PREV_AMT_APPLICATION_MEAN': float(request.form.get('PREV_AMT_APPLICATION_MEAN')),
                'PREV_AMT_GOODS_PRICE_MEDIAN': float(request.form.get('PREV_AMT_GOODS_PRICE_MEDIAN')),
                'HOUR_APPR_PROCESS_START': float(request.form.get('HOUR_APPR_PROCESS_START')),
                'PREV_AMT_GOODS_PRICE_MEAN': float(request.form.get('PREV_AMT_GOODS_PRICE_MEAN')),
                'PREV_YEARS_FIRST_DRAWING_MEAN': float(request.form.get('PREV_YEARS_FIRST_DRAWING_MEAN')),
                'PREV_SK_ID_CURR_FIRST': float(request.form.get('PREV_SK_ID_CURR_FIRST')),
                'OBS_60_CNT_SOCIAL_CIRCLE': float(request.form.get('OBS_60_CNT_SOCIAL_CIRCLE')),
                'PREV_NAME_GOODS_CATEGORY_<LAMBDA>': float(request.form.get('PREV_NAME_GOODS_CATEGORY_<LAMBDA>')),
                'PREV_NFLAG_INSURED_ON_APPROVAL_MAX': float(request.form.get('PREV_NFLAG_INSURED_ON_APPROVAL_MAX')),
                'WEEKDAY_APPR_PROCESS_START': request.form.get('WEEKDAY_APPR_PROCESS_START'),
                'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>': float(request.form.get('PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>'))
            }
            
            final_input = predictor.convert_to_model_format(user_data)
            prob, is_default, error = predictor.predict(final_input)
            
            if error:
                return render_template('result.html', error=error)
                
            return render_template('result.html', prob=round(prob*100, 2), is_default=is_default)

        except Exception as e:
            return render_template('result.html', error=f"Input Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)