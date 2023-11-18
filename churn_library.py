'''
The churn_library.py is a library of functions to find customers who are likely to churn.
'''

# import libraries
import os
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import CATEGORICAL_COLUMNS
from constants import KEEP_COLS
from constants import QUANT_COLUMNS
from constants import DATA_FILE_PATH
from constants import SPECIFIC_COLUMN
from constants import SPECIFIC_COLUMN_ELEMENT
from constants import TARGET_LABEL
from constants import IMAGE_RESULTS_PATH_RF_RESULTS
from constants import IMAGE_RESULTS_PATH_LOGISTIC_RESULTS
from constants import IMAGE_RESULTS_PATH_ROC_CURVE_RESULT
from constants import IMAGE_RESULTS_FEATURE_IMPORTANCES_NAME
from constants import IMAGE_PATH_RESULTS
from constants import IMAGE_EDA_PATH_CHURN_DISTRIBUTION_NAME
from constants import IMAGE_EDA_PATH_CUSTOMER_AGE_DISTRIBUTION
from constants import IMAGE_EDA_PATH_EDA_MATERIAL_STATUS_DISTRIBUTION
from constants import IMAGE_EDA_PATH_EDA_TOTAL_TRANSACTION_DISTRIBUTION
from constants import IMAGE_EDA_PATH_HEATMAP
from constants import MODEL_PATH_LOGISTIC
from constants import MODEL_PATH_RFC
from constants import PARAMERTER_GRID

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_imported = pd.read_csv(pth, index_col=0)
    df_imported[TARGET_LABEL] = df_imported[SPECIFIC_COLUMN].apply(
        lambda val: 0 if val == SPECIFIC_COLUMN_ELEMENT else 1
    )
    df_imported = df_imported.drop([SPECIFIC_COLUMN], axis=1)

    return df_imported


def perform_eda(df_imported):
    '''
    perform eda on df and save figures to images folder
    input:
            df_imported: pandas dataframe

    output:
            None
    '''
    df_edf = df_imported.copy(deep=True)
    #
    plt.figure(figsize=(20, 10))
    df_edf[TARGET_LABEL].hist()
    plt.savefig(IMAGE_EDA_PATH_CHURN_DISTRIBUTION_NAME)
    plt.close()
    #
    plt.figure(figsize=(20, 10))
    df_edf['Customer_Age'].hist()
    plt.savefig(IMAGE_EDA_PATH_CUSTOMER_AGE_DISTRIBUTION)
    plt.close()
    #
    plt.figure(figsize=(20, 10))
    df_edf.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(IMAGE_EDA_PATH_EDA_MATERIAL_STATUS_DISTRIBUTION)
    plt.close()
    #
    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df_edf['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(IMAGE_EDA_PATH_EDA_TOTAL_TRANSACTION_DISTRIBUTION)
    plt.close()
    #
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df_edf[QUANT_COLUMNS].corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(IMAGE_EDA_PATH_HEATMAP)
    plt.close()

    # Return dataframe
    return df_edf


def encoder_helper(df_imported, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df_imported: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df_encoder = df_imported.copy(deep=True)
    for category in category_lst:
        list_column = []
        column_groups = df_imported.groupby(
            category).agg({TARGET_LABEL: 'mean'})
        for val in df_imported[category]:
            list_column.append(column_groups.loc[val])
        if response:
            df_encoder[category + '_' + response] = list_column
        else:
            df_encoder[category] = list_column
    # Return dataframe
    return df_encoder


def perform_feature_engineering(df_imported, response):
    '''
    input:
              df_imported: pandas dataframe
              response: string of response name [optional argument that could be used
                  for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # feature engineering
    df_encoded = encoder_helper(
        df_imported,
        CATEGORICAL_COLUMNS,
        response=response
    )
    # target feature
    y = df_encoded[TARGET_LABEL]
    # Create dataFrame
    X = pd.DataFrame()
    # Features dataFrame
    X[KEEP_COLS] = df_encoded[KEEP_COLS]
    for feature in X.columns:
        X[feature] = X[feature].astype(np.float16)
    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results
        and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # LogisticRegression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(IMAGE_RESULTS_PATH_LOGISTIC_RESULTS)
    plt.close()

    # RandomForestClassifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(IMAGE_RESULTS_PATH_RF_RESULTS)
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(30, 5))
    # Create plot title
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    # Save the image
    plt.savefig(fname=output_pth + IMAGE_RESULTS_FEATURE_IMPORTANCES_NAME)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)
    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAMERTER_GRID, cv=5)
    cv_rfc.fit(X_train, y_train)
    # LogisticRegression
    lrc.fit(X_train, y_train)
    # Save best models
    joblib.dump(cv_rfc.best_estimator_, MODEL_PATH_RFC)
    joblib.dump(lrc, MODEL_PATH_LOGISTIC)
    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(
        lrc, X_test, y_test, ax=axis, alpha=0.8)
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(IMAGE_RESULTS_PATH_ROC_CURVE_RESULT)
    plt.close()
    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)
    # Compute and feature importance
    feature_importance_plot(
        model=cv_rfc,
        X_data=X_test,
        output_pth=IMAGE_PATH_RESULTS
    )


if __name__ == "__main__":
    # Import data
    df = import_data(DATA_FILE_PATH)
    # Perform EDA
    df_eda = perform_eda(df)
    # Feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_eda, response=TARGET_LABEL
    )
    # Model training, prediction and evaluation
    train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
