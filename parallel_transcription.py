import requests
import concurrent.futures
import random
import string
# Define the API endpoint and any parameters
base_url = "http://18.205.156.240:8000/transcribe"
# Define the function to make the API call
def make_api_call():
    def generate_random_string(length):
        characters = string.ascii_letters + string.digits  # You can customize this to include other characters
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string
    params = {
        "diarize": True,
        "request_id":generate_random_string(6) ,
        "language": "en",
        "sequence_id": 0,
        "max_num_speakers": 2,
        "min_num_speakers": 1
    }
    base_url = "http://18.205.156.240:8000/transcribe"
    t_url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
    files=[
        ('audio_file',('file',open(f"/Users/I1597/Downloads/5sec.wav",'rb'),'application/octet-stream'))
    ]
    response = requests.request("POST", t_url, headers={}, data={}, files=files)
    res = response.json()
    print(res)

# Number of parallel requests
num_requests = 200

# Create a ThreadPoolExecutor with max workers
with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
    # Submit API calls to the executor and store the future objects
    futures = [executor.submit(make_api_call) for _ in range(num_requests)]

# Collect and process the results when the API calls are completed
results = [future.result() for future in concurrent.futures.as_completed(futures)]

# Now 'results' contains the responses from all 200 API calls
