from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG("data_pipeline", start_date=datetime(2026, 1, 1), schedule=None, catchup=False, tags=["ecoai"]) as dag:
    BashOperator(task_id="build_manifest",
                 bash_command="cd /opt/airflow/project && python -m src.data_prep.prepare_data --config configs/data.yaml")
