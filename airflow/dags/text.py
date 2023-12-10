from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    'owner': 'Admin',
    'start_date': dt.datetime(2023, 12, 1),
    'retries': 1,
    'retry_delays': dt.timedelta(minutes=1),
    'depends_on_past': False,
    'provide_context': True
}

with DAG(
    dag_id='KOH',
    default_args=args,
    schedule_interval=None,
    tags=['google', 'score'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="cd /home/kda/ML_lab3/MLops_lab3/datasets && gdown 1X-mGhWO8DwPLljP1e0gkdAloD-GL8SLq", 
                            dag=dag)

    data_1 = BashOperator(task_id='data_1',
                            bash_command="python /home/kda/ML_lab3/MLops_lab3/scripts/script1.py", 
                            dag=dag)
    data_2 = BashOperator(task_id='data_2',
                            bash_command="python /home/kda/ML_lab3/MLops_lab3/scripts/script2.py", 
                            dag=dag)
    data_3 = BashOperator(task_id='data_3',
                            bash_command="python /home/kda/ML_lab3/MLops_lab3/scripts/script3.py", 
                            dag=dag)
    data_tts = BashOperator(task_id='data_tts',
                            bash_command="cd /home/kda/ML_lab3/MLops_lab3/ && python /home/kda/ML_lab3/MLops_lab3/scripts/train_test_split.py", 
                            dag=dag)
    
    data_ml = BashOperator(task_id='data_ml',
                            bash_command="cd /home/kda/ML_lab3/MLops_lab3/ && python /home/kda/ML_lab3/MLops_lab3/scripts/model_learn.py", 
                            dag=dag)

    get_data >> data_1 >> data_2 >> data_3 >> data_tts >> data_ml
