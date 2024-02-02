"""
Step 1: Load the Model

Note: 
1. This won't be a mandatory step, This is required when you want to work with your own configurations
and with different model size other than the default one.
The current default model is base which will change later. Foe this QA session we will have to load models because the default
model i.e. base do not provide a good transcription. 

2. If you use same configurations, no new model will be loaded, rather the same model will be reused. 
while testing try loading as many models you want with same configurations and keep deleting it.
Keep the count of creation and deletion. if deletion count is greater, you'll get a message in response while deleting that the key
doesn't exist.
"""

### Optional API

import requests

beam_size = 1  # any integer value, preferrably 5 or below. Higher the value, slower the performance but better the transcription. default: 1
temperature = 0.0 # any float value between 0 and 1 i.e. 0.2, 0.4, 0.6 etc. Lower temperatures tries to keeps the output constant default: 0.0
no_speech_threshold = 0.6 # any float value between 0 to 1. Lower the value, low chances of detecting silences. default 0.6
model_size = "base" # whisper x model to be used. options in increasing order of size -> tiny, base, medium, large-v1, large-v2. default base. Larger the model better the transcription and longer the transcription duration. 
# try to use model large-v2, large-v1 or medium only. We'll most probably be using large-v2 in prod.

# Load the model
base_url = "http://18.205.156.240:8000/load_model"
params = {
    "beam_size": beam_size,
    "temperature": temperature,
    "no_speech_threshold": no_speech_threshold,
    "model_size": model_size
}

init_url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
init_response = requests.request("POST", init_url, headers={}, data={})
model_key = init_response.json().get("model_key")
print("model", model_key)


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



"""
Step 2.1: Full Transcription and Diarization
"""

import requests
import time

# Transcription

diarize = True # whether you want diarization default: False
request_id = "1111111" # any random string
sequence_id = 0 # optional, This will be used with chunk diarization, for now this can be kept as zero or ignored
audio_path = "" # path of the audio file
language = "en" # Optional parameter, if provided, transcription will be a little faster
max_num_speakers = 2 # optional, if provided, diarization will have a little improvement
min_num_speakers = 1 # optional, if provided, diarization will have a little improvement default: 1

base_url = "http://18.205.156.240:8000/transcribe"
params = {
    "diarize": diarize,
    "request_id": request_id,
    "model_key": model_key,
    "sequence_id": sequence_id,
    "language": language,
    "max_num_speakers": max_num_speakers,
    "min_num_speakers": min_num_speakers
}

t_url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
files=[
    ('audio_file',('file',open(audio_path,'rb'),'application/octet-stream'))
]
t1 = time.time()
response = requests.request("POST", t_url, headers={}, data={}, files=files)
t2 = time.time()

print("Request Time", (t2-t1))
print(response.json())




#####################################################################################################################
#####################################################################################################################



"""
Step 2.2: Chunk Transcription and Diarization
"""

# Audios with length < 15 will be considered for chunk diarization
# Transcription
import requests
import time
from pydub import AudioSegment

diarize = True # whether you want diarization default: False
request_id = "1111111" # any random string
sequence_id = 0 # optional, This will be used with chunk diarization, for now this can be kept as zero or ignored
audio_path = "/Users/I1597/Downloads/Data/Audio Recordings/CAR0001.mp3" # path of the audio file
language = "en" # Optional parameter, if privided, transcription will be a little faster
max_num_speakers = 2 # optional, if provided, diarization will have a little improvement
min_num_speakers = 1 # optional, if provided, diarization will have a little improvement
chunk_folder_path = "/Users/I1597/Downloads/CAR0001"
audio_format = "mp3"
base_url = "http://18.205.156.240:8000/transcribe"

audio = AudioSegment.from_file(audio_path, format=audio_format)
chunk_duration = 5 * 1000  # 5 seconds in milliseconds
chunks = [audio[i:i + chunk_duration] for i in range(0, len(audio), chunk_duration)]
for i, chunk in enumerate(chunks):
    chunk.export(f"{chunk_folder_path}/chunk_{i}.{audio_format}", format=audio_format)

result = []
for i in range(10):
    t1 = time.time()
    params = {
        "diarize": diarize,
        "request_id": request_id,
        "language": language,
        "sequence_id": i,
        "max_num_speakers": max_num_speakers,
        "min_num_speakers": min_num_speakers
    }
    t_url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
    print(i)
    files=[
        ('audio_file',('file',open(f"{chunk_folder_path}/chunk_{i}.{audio_format}",'rb'),'application/octet-stream'))
    ]
    response = requests.request("POST", t_url, headers={}, data={}, files=files)
    res = response.json()
    result.append(res)
    print(res)
    t2 = time.time()
    print("total time", (t2-t1))
    print("*"*100)

print("#"*50, "Final Result", "#"*50)
print()
print(result)
print()
print("*"*112)


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################




"""
Step 2.3 Send s3 file path for diarization
"""

import requests
import time

# Transcription

diarize = True # whether you want diarization default: False
request_id = "1111111" # any random string
sequence_id = 0 # optional, This will be used with chunk diarization, for now this can be kept as zero or ignored
audio_path = "s3://{bucket_name}/{file_path}" # path of the audio file
language = "en" # Optional parameter, if provided, transcription will be a little faster
max_num_speakers = 2 # optional, if provided, diarization will have a little improvement
min_num_speakers = 1 # optional, if provided, diarization will have a little improvement default: 1

base_url = "http://18.205.156.240:8000/transcribe"
params = {
    "diarize": diarize,
    "request_id": request_id,
    "model_key": model_key,
    "sequence_id": sequence_id,
    "language": language,
    "max_num_speakers": max_num_speakers,
    "min_num_speakers": min_num_speakers
}

t_url = f"{base_url}?{'&'.join(f'{key}={value}' for key, value in params.items())}"
t1 = time.time()
response = requests.request("POST", t_url, headers={}, data={})
t2 = time.time()
print("Response Time", (t2-t1))
print(response.json())

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################




"""
Step 3: Delete the model
"""

# Delete loaded model
# This is required before loading another model because the model takes huge space and 
# currently we can not afford to run two large size models on the GPU

model_key = "model key" # the model key that was received while loading the model
delete_url = f"http://18.205.156.240:8000/model/{model_key}"
response = requests.delete(delete_url)
print(response.json())


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


"""
Step 4: Delete Speaker segments files (Only in case of chunk diarization)
"""

delete_url = f"http://18.205.156.240:8000/transcribe/{request_id}"
response = requests.delete(delete_url)
print(response.json())

