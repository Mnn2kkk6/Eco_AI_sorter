import os
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

HOST = os.environ["HOST_PROJECT_DIR"]  # đường dẫn dự án trên máy host (đặt trong .env)

with DAG("train_pipeline", start_date=datetime(2026, 1, 1), schedule=None, catchup=False, tags=["ecoai"]) as dag:
    DockerOperator(
        task_id="train_multitask", image="ecoai-training:latest", auto_remove="success",
        command="python -m src.train --epochs 15",
        docker_url="unix://var/run/docker.sock", network_mode="bridge", mount_tmp_dir=False,
        mounts=[Mount("/app/data", f"{HOST}/data", type="bind"), Mount("/app/weights", f"{HOST}/weights", type="bind")],
        # có GPU: device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
    )
