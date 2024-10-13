# kafka_scripts/kafka_producer.py

from kafka import KafkaProducer
import json
import time
import pandas as pd

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

def get_data():
    # Read data from a source, e.g., CSV file or database
    url = 'https://raw.githubusercontent.com/richardcornall/Tensorflow-Pytorch-Ensemble-Machine-Learning-Model-for-Breast-Cancer-Detection-/main/data.csv'
    data = pd.read_csv(url)
    data = data.dropna(axis=1)
    data = data.to_dict(orient='records')
    return data

if __name__ == '__main__':
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                             value_serializer=json_serializer)
    data = get_data()
    for record in data:
        producer.send('breast_cancer_data', record)
        time.sleep(1)  # Sleep to simulate real-time data ingestion
