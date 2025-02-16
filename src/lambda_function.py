import json
import boto3
import joblib
import pandas as pd
import os
from pygam import LinearGAM, s
from io import BytesIO

import logging


def load_model_from_s3():
    # Retrieve environment variables
    try:
        # Configurar el cliente S3
        access_key_id = os.environ.get('AWS_ACCESS_KEYID')
        secret_access_key = os.environ.get('AWS_SECRET_ACCESSKEY')
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="us-east-2"
        )

        s3_client = session.client('s3')
        bucket_name = "homerunscientist2024"

        object_key = "linear_gam_model.joblib"

        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        model_content = response['Body'].read()
        model = joblib.load(BytesIO(model_content))
        logging.info("Model loaded successfully.")
        logging.info("<<<< ok getting the model from S3 >>>>")
        return model, "Success"
    except Exception as e:
        logging.error(e)
        return False, str(3)


# Function to classify speed based on the prediction
def class_spped(x):
    if x > 146:
        return 'Excellent'
    elif x >= 135 and x < 146:
        return 'Good'
    elif x >= 116 and x < 135:
        return 'Fair'
    elif x >= 96 and x < 116:
        return 'So-so'
    else:
        return 'Poor'



import json
import logging
import pandas as pd

def lambda_handler(event, context):
    msj = ""
    try:
        gam, msj = load_model_from_s3()
    except Exception as e:
        logging.error(e)
        gam, msj = False, str(e)
        logging.info("here -------------->>>", event, str(msj))
    try:
        if gam is False:
            msj = "GAM not found"
            logging.error(msj)
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'prediction': -99,
                    'class': msj
                })
            }

        logging.info(f"Received data: {event}")
        msj = "GAM found"

        if 'body' not in event or not event['body']:
            msj = "body is not valid"
            logging.error(msj)
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'prediction': -99,
                    'class': msj
                })
            }
    except Exception as e:
        logging.error(f"Exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'prediction': -99,
                'class': str(e)
            })
        }

    try:
        parsed_body = event['body']

        relative_distance_from_handle_to_hit = parsed_body['relative_distance_from_handle_to_hit']
        swing_velocity_head_hit = parsed_body['swing_velocity_head_hit']
        swing_displacement_handle_hit = parsed_body['swing_displacement_handle_hit']
        hit_spray_angle = parsed_body['hit_spray_angle']
        hit_launch_angle = parsed_body['hit_launch_angle']
        # Preparing
        result = {
            "relative_distance_from_handle_to_hit": [relative_distance_from_handle_to_hit],
            "swing_velocity_head_hit": [swing_velocity_head_hit],
            "swing_displacement_handle_hit": [swing_displacement_handle_hit],
            "hit_spray_angle": [hit_spray_angle],
            "hit_launch_angle": [hit_launch_angle]
        }

        #  DataFrame
        X_test_single = pd.DataFrame(result)

    except KeyError as e:
        msj = f"Missing key in JSON: {str(e)} + {event['body']}"
        logging.error(msj)
        return {
                'statusCode': 400,
                'body': json.dumps({
                    'prediction': -99,
                    'class': msj
                })
            }
    try:

        # Prediction and classification
        prediction_single = gam.predict(X_test_single)

        print(f'Prediction: {prediction_single[0]}')
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction_single[0],
                'class': class_spped(prediction_single[0])
            })
        }

    except Exception as e:
        logging.error(f"Exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'prediction': -99,
                'class': str(e)
            })
        }
