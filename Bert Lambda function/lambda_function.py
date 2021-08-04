import json

import boto3

def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')
    texT = '' 
    try : 
        texT = eval(event['body'])['body'].lower()
    except :
        texT = event['body'].lower()
    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = 'tensorflow-inference-2021-07-25-10-51-04-285',    # The name of the endpoint we created
                                       ContentType = 'text/plain',                 # The data format that is expected
                                       Body = texT)                       # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8') 
    print(str(result))
    result =int(eval(result)['predictions'][0][0] > 0.75)
    print(event['body'])
    print(str(result))
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }