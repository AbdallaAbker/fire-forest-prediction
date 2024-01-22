import yaml
import argparse
import pandas as pd
import joblib

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def model_monitoring(config_path):
    config = read_params(config_path)
    train_data_path = config["raw_data_config"]["raw_data_csv"]
    new_train_data_path = config["raw_data_config"]["new_train_data_csv"]
    target = config["raw_data_config"]["target"]
    monitor_dashboard_path = config["model_monitor"]["monitor_dashboard_html"]
    monitor_target = config["model_monitor"]["target_col_name"]
    model_dir_path = config["model_webapp_dir"]
    num_features = config["raw_data_config"]["num_features"]

    try:
        ref = pd.read_csv(train_data_path)
        cur = pd.read_csv(new_train_data_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Load model
    model = joblib.load(model_dir_path)

    # Ensure the dataframe is not just a view or copy
    ref = ref.copy()
    cur = cur.copy()

    # Rename columns
    ref = ref.rename(columns={target: monitor_target}, inplace=False)
    cur = cur.rename(columns={target: monitor_target}, inplace=False)

    # Generate predictions
    train_preds = model.predict(ref[num_features])
    ref['prediction'] = train_preds
    val_preds = model.predict(cur[num_features])
    cur['prediction'] = val_preds

    # Define column mapping
    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=None
    )

    # Generate Drift and Missing Values Report
    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)
    report.save(monitor_dashboard_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_monitoring(config_path=parsed_args.config)
