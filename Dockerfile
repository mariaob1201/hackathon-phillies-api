FROM public.ecr.aws/lambda/python:3.10

COPY src/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src/lambda_function.py app.py

CMD [ "app.lambda_handler" ]
