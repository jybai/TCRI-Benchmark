'''
https://stackoverflow.com/a/39225039
'''
import requests
import argparse
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    pretraind_models = {
        'ConvLSTM.zip': '1ZEUfZXY00eFcbCcZ6W8pen77xDm8rOSB',
        'ConvLSTM_cca.zip': '1UZKRqG5hpNLT-U-DkJJV4kG3T328gDi1',
        'ConvLSTM_ssa.zip': '1JAV_ZiMdiArnqhnU0S3HBRb_vLmpm4G2',
        'ConvLSTM_cca_ssa.zip': '1BYh5t_JlYGT4QqHPP_QkahDgLq3PIXNO',
    }

    for dest, file_id in tqdm(pretraind_models.items()):
        download_file_from_google_drive(file_id, dest)

