# fastapi_keras_docker
Uses DVC

To get an inference, run the container and use the endpoint /prediction/.

With for example postman, use POST in the BODY use form-data and use file and choose an input image.

## Setup

To get started, create a virtual environment with either with [pipenv][1] or venv:

    pipenv install -d
    pipenv shell
    
    OR
    
    python3 -m venv .env
    source .env/bin/activate
    pip install -r requirements.txt

Pull the trained model via DVC via the read-only or full access endpoint:

    dvc pull

    OR

    dvc remote add --local -d ph-ofai-public-s3 s3://ofai-public/dvc/
    dvc remote modify --local ph-ofai-public-s3 endpointurl https://s3.protohaus.org/
    dvc remote modify --local ph-ofai-public-s3 access_key_id <access-key-id>
    dvc remote modify --local ph-ofai-public-s3 secret_access_key <secret_access_key>
    dvc pull

To start the app after having loaded the virtual environment:

    python app/main.py

Use the following command to request an inference on an image:

    curl --request POST 'http://localhost:8001/prediction/' --form 'file=@"/path/to/image.jpg"'

[1]: https://pipenv.pypa.io/en/latest/