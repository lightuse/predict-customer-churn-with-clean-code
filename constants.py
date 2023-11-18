'''
This is the constants.py procedure.
'''

LOG_FILE_PATH = './logs/churn_library.log'
DATA_FILE_PATH = './data/bank_data.csv'

SPECIFIC_COLUMN = 'Attrition_Flag'
SPECIFIC_COLUMN_ELEMENT = 'Existing Customer'

TARGET_LABEL = 'Churn'

IMAGE_PATH_EDA = './images/eda/'
IMAGE_EDA_CHURN_DISTRIBUTION_NAME = 'churn_distribution.png'
IMAGE_EDA_PATH_CHURN_DISTRIBUTION_NAME = IMAGE_PATH_EDA + IMAGE_EDA_CHURN_DISTRIBUTION_NAME
IMAGE_EDA_CUSTOMER_AGE_DISTRIBUTION_NAME = 'customer_age_distribution.png'
IMAGE_EDA_PATH_CUSTOMER_AGE_DISTRIBUTION = IMAGE_PATH_EDA + IMAGE_EDA_CUSTOMER_AGE_DISTRIBUTION_NAME
IMAGE_EDA_MATERIAL_STATUS_DISTRIBUTION_NAME = 'marital_status_distribution.png'
IMAGE_EDA_PATH_EDA_MATERIAL_STATUS_DISTRIBUTION = IMAGE_PATH_EDA + IMAGE_EDA_MATERIAL_STATUS_DISTRIBUTION_NAME
IMAGE_EDA_TOTAL_TRANSACTION_DISTRIBUTION_NAME = 'total_transaction_distribution.png'
IMAGE_EDA_PATH_EDA_TOTAL_TRANSACTION_DISTRIBUTION = IMAGE_PATH_EDA + IMAGE_EDA_TOTAL_TRANSACTION_DISTRIBUTION_NAME
IMAGE_EDA_HEATMAP_NAME = 'heatmap.png'
IMAGE_EDA_PATH_HEATMAP = IMAGE_PATH_EDA + IMAGE_EDA_HEATMAP_NAME

IMAGE_PATH_RESULTS = './images/results/'
IMAGE_RESULTS_RF_RESULTS_NAME = 'rf_results.png'
IMAGE_RESULTS_PATH_RF_RESULTS = IMAGE_PATH_RESULTS + IMAGE_RESULTS_RF_RESULTS_NAME
IMAGE_RESULTS_LOGISTIC_RESULTS_NAME = 'logistic_results.png'
IMAGE_RESULTS_PATH_LOGISTIC_RESULTS = IMAGE_PATH_RESULTS + IMAGE_RESULTS_LOGISTIC_RESULTS_NAME
IMAGE_RESULTS_ROC_CURVE_RESULT_NAME = 'roc_curve_result.png'
IMAGE_RESULTS_PATH_ROC_CURVE_RESULT = IMAGE_PATH_RESULTS + IMAGE_RESULTS_ROC_CURVE_RESULT_NAME
IMAGE_RESULTS_FEATURE_IMPORTANCES_NAME = 'feature_importances.png'
IMAGE_RESULTS_PATH_FEATURE_IMPORTANCES = IMAGE_PATH_RESULTS + IMAGE_RESULTS_FEATURE_IMPORTANCES_NAME

MODEL_PATH_LOGISTIC_NAME = 'logistic_model.pkl'
MODEL_PATH_LOGISTIC = './models/' + MODEL_PATH_LOGISTIC_NAME
MODEL_PATH_RFC_NAME = 'rfc_model.pkl'
MODEL_PATH_RFC = './models/' + MODEL_PATH_RFC_NAME

CATEGORICAL_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn'
]

PARAMERTER_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
