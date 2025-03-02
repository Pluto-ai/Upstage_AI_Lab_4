from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import papermill as pm
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_eda_ipynb_workflow',
    default_args=default_args,
    description='ML and EDA workflow with .ipynb files and S3 integration',
    schedule_interval=timedelta(days=1),
)

def download_from_s3(key, bucket_name, local_path):
    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.download_file(key=key, bucket_name=bucket_name, local_path=local_path)

def upload_to_s3(local_path, key, bucket_name):
    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_file(filename=local_path, key=key, bucket_name=bucket_name, replace=True)

def run_notebook(input_path, output_path, parameters=None):
    pm.execute_notebook(
        input_path,
        output_path,
        parameters=parameters or {}
    )

def start_fastapi_server():
    from subprocess import Popen
    Popen(["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"])

with dag:
    download_data = PythonOperator(
        task_id='download_data',
        python_callable=download_from_s3,
        op_kwargs={
            'key': 'data/final_data.csv',
            'bucket_name': 'hyunaebucket',
            'local_path': '/opt/airflow/data/final_data.csv'
        }
    )

    run_eda = PythonOperator(
        task_id='run_eda',
        python_callable=run_notebook,
        op_kwargs={
            'input_path': '/opt/airflow/notebooks/eda.ipynb',
            'output_path': '/opt/airflow/notebooks/eda_output.ipynb',
            'parameters': {
                'input_data_path': '/opt/airflow/data/final_data.csv',
                'output_path': '/opt/airflow/data/eda_results.csv'
            }
        }
    )

    train_ml_model = PythonOperator(
        task_id='train_ml_model',
        python_callable=run_notebook,
        op_kwargs={
            'input_path': '/opt/airflow/notebooks/train_model.ipynb',
            'output_path': '/opt/airflow/notebooks/train_model_output.ipynb',
            'parameters': {
                'input_data_path': '/opt/airflow/data/final_data.csv',
                'model_output_path': '/opt/airflow/models/model.joblib'
            }
        }
    )

    upload_eda_results = PythonOperator(
        task_id='upload_eda_results',
        python_callable=upload_to_s3,
        op_kwargs={
            'local_path': '/opt/airflow/data/eda_results.csv',
            'key': 'results/eda_results.csv',
            'bucket_name': 'hyunaebucket'
        }
    )

    upload_ml_model = PythonOperator(
        task_id='upload_ml_model',
        python_callable=upload_to_s3,
        op_kwargs={
            'local_path': '/opt/airflow/models/model.joblib',
            'key': 'models/model.joblib',
            'bucket_name': 'hyunaebucket'
        }
    )

    start_api = PythonOperator(
        task_id='start_fastapi_server',
        python_callable=start_fastapi_server
    )

    download_data >> [run_eda, train_ml_model] >> [upload_eda_results, upload_ml_model] >> start_api