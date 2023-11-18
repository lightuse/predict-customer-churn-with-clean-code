'''
Testing module that will check the churn_library.py procedure.
Artifact produced will be in logs folders.
'''

from math import ceil
import logging
import pathlib
import churn_library as cls

from constants import LOG_FILE_PATH
from constants import DATA_FILE_PATH
from constants import CATEGORICAL_COLUMNS
from constants import TARGET_LABEL

from constants import MODEL_PATH_LOGISTIC_NAME
from constants import MODEL_PATH_LOGISTIC
from constants import MODEL_PATH_RFC_NAME
from constants import MODEL_PATH_RFC

from constants import IMAGE_EDA_HEATMAP_NAME
from constants import IMAGE_EDA_CHURN_DISTRIBUTION_NAME
from constants import IMAGE_EDA_CUSTOMER_AGE_DISTRIBUTION_NAME
from constants import IMAGE_EDA_MATERIAL_STATUS_DISTRIBUTION_NAME
from constants import IMAGE_EDA_TOTAL_TRANSACTION_DISTRIBUTION_NAME
from constants import IMAGE_RESULTS_PATH_FEATURE_IMPORTANCES
from constants import IMAGE_RESULTS_PATH_RF_RESULTS
from constants import IMAGE_RESULTS_PATH_LOGISTIC_RESULTS
from constants import IMAGE_RESULTS_PATH_ROC_CURVE_RESULT
from constants import IMAGE_RESULTS_FEATURE_IMPORTANCES_NAME
from constants import IMAGE_RESULTS_LOGISTIC_RESULTS_NAME
from constants import IMAGE_RESULTS_RF_RESULTS_NAME
from constants import IMAGE_RESULTS_ROC_CURVE_RESULT_NAME
from constants import IMAGE_EDA_PATH_CHURN_DISTRIBUTION_NAME
from constants import IMAGE_EDA_PATH_CUSTOMER_AGE_DISTRIBUTION
from constants import IMAGE_EDA_PATH_EDA_MATERIAL_STATUS_DISTRIBUTION
from constants import IMAGE_EDA_PATH_EDA_TOTAL_TRANSACTION_DISTRIBUTION
from constants import IMAGE_EDA_PATH_HEATMAP

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def has_file(image_path):
    '''
    check file exsit 
    input:
            image_path: image flie path
    output:
            TRUE: exist
            FALSE: not exist
    '''
    path = pathlib.Path(image_path)

    return path.exists()


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_imported = cls.import_data(DATA_FILE_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_imported.shape[0] > 0
        assert df_imported.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df_imported = cls.import_data(DATA_FILE_PATH)

    try:
        cls.perform_eda(df_imported)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logging.error('Column "%s" not found', err.args[0])
        raise err
    # Assert if `churn_distribution.png` is created
    try:
        assert has_file(IMAGE_EDA_PATH_CHURN_DISTRIBUTION_NAME) is True
        logging.info('File %s was found', IMAGE_EDA_CHURN_DISTRIBUTION_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `customer_age_distribution.png` is created
    try:
        assert has_file(IMAGE_EDA_PATH_CUSTOMER_AGE_DISTRIBUTION) is True
        logging.info(
            'File %s was found',
            IMAGE_EDA_CUSTOMER_AGE_DISTRIBUTION_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `marital_status_distribution.png` is created
    try:
        assert has_file(IMAGE_EDA_PATH_EDA_MATERIAL_STATUS_DISTRIBUTION) is True
        logging.info(
            'File %s was found',
            IMAGE_EDA_MATERIAL_STATUS_DISTRIBUTION_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `total_transaction_distribution.png` is created
    try:
        assert has_file(IMAGE_EDA_PATH_EDA_TOTAL_TRANSACTION_DISTRIBUTION) is True
        logging.info(
            'File %s was found',
            IMAGE_EDA_TOTAL_TRANSACTION_DISTRIBUTION_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `heatmap.png` is created
    try:
        assert has_file(IMAGE_EDA_PATH_HEATMAP) is True
        logging.info('File %s was found', IMAGE_EDA_HEATMAP_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    # Load DataFrame
    df_imported = cls.import_data(DATA_FILE_PATH)

    try:
        df_encoded = cls.encoder_helper(
            df_imported,
            category_lst=[],
            response=None)
        # Data should be the same
        assert df_encoded.equals(df_imported) is True
        logging.info(
            "Testing encoder_helper(data_frame, category_lst=[]): SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(data_frame, category_lst=[]): ERROR"
        )
        raise err

    try:
        df_encoded = cls.encoder_helper(
            df_imported,
            category_lst=CATEGORICAL_COLUMNS,
            response=None
        )
        # Column names should be same
        assert df_encoded.columns.equals(df_imported.columns) is True
        # Data should be different
        assert df_encoded.equals(df_imported) is False
        logging.info(
            "Testing encoder_helper(df, cat_columns, response=None): SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(df, cat_columns, response=None): ERROR"
        )
        raise err

    try:
        df_encoded = cls.encoder_helper(
            df_imported,
            category_lst=CATEGORICAL_COLUMNS,
            response=TARGET_LABEL
        )
        # Columns names should be different
        assert df_encoded.columns.equals(df_imported.columns) is False
        # Data should be different
        assert df_encoded.equals(df_imported) is False
        # Number of columns in encoded_df is the sum of columns in data_frame
        # and the newly created columns from cat_columns
        assert len(df_encoded.columns) == len(df_imported.columns) + len(CATEGORICAL_COLUMNS)
        logging.info(
            "Testing encoder_helper(df, cat_columns, response='Churn'): SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(df, cat_columns, response='Churn'): ERROR"
        )
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    # Load the DataFrame
    df_imported = cls.import_data(DATA_FILE_PATH)

    try:
        (_, X_test, _, _) = cls.perform_feature_engineering(
            df_imported, response=TARGET_LABEL
        )
        # `Churn` must be present in `dataframe`
        assert TARGET_LABEL in df_imported.columns
        logging.info(
            "Testing perform_feature_engineering. `Churn` column is present: SUCCESS"
        )
    except KeyError as err:
        logging.error(
            'The `Churn` column is not present in the DataFrame: ERROR'
        )
        raise err

    try:
        # X_test size should be 30% of `dataframe`
        assert (
            X_test.shape[0] == ceil(df_imported.shape[0] * 0.3)
        ) is True
        logging.info(
            'Testing perform_feature_engineering. DataFrame sizes are consistent: SUCCESS'
        )
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering. DataFrame sizes are not correct: ERROR'
        )
        raise err


def test_train_models():
    '''
    test train_models
    '''
    # Load the DataFrame
    df_imported = cls.import_data(DATA_FILE_PATH)
    # Feature engineering
    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
        df_imported,
        response=TARGET_LABEL
    )
    # Assert if `logistic_model.pkl` file is present
    try:
        cls.train_models(X_train, X_test, y_train, y_test)
        assert has_file(MODEL_PATH_LOGISTIC) is True
        logging.info('File %s was found', MODEL_PATH_LOGISTIC_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `rfc_model.pkl` file is present
    try:
        assert has_file(MODEL_PATH_RFC) is True
        logging.info('File %s was found', MODEL_PATH_RFC_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `roc_curve_result.png` file is present
    try:
        assert has_file(IMAGE_RESULTS_PATH_ROC_CURVE_RESULT) is True
        logging.info('File %s was found', IMAGE_RESULTS_ROC_CURVE_RESULT_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `rfc_results.png` file is present
    try:
        assert has_file(IMAGE_RESULTS_PATH_RF_RESULTS) is True
        logging.info('File %s was found', IMAGE_RESULTS_RF_RESULTS_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `logistic_results.png` file is present
    try:
        assert has_file(IMAGE_RESULTS_PATH_LOGISTIC_RESULTS) is True
        logging.info('File %s was found', IMAGE_RESULTS_LOGISTIC_RESULTS_NAME)
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err
    # Assert if `feature_importances.png` file is present
    try:
        assert has_file(IMAGE_RESULTS_PATH_FEATURE_IMPORTANCES) is True
        logging.info(
            'File %s was found',
            IMAGE_RESULTS_FEATURE_IMPORTANCES_NAME
        )
    except AssertionError as err:
        logging.error('Not such file on disk')
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
