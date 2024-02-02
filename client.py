# import socketio

# sio = socketio.Client()

# @sio.event
# def connect():
#     print("Connected to the server")

# @sio.event
# def message(data):
#     print(f"Server says: {data}")

# @sio.event
# def disconnect():
#     print("Disconnected from the server")

# if __name__ == "__main__":
#     sio.connect("http://0.0.0.0:8000")
#     while True:
#         user_input = input("Enter a message (or 'exit' to quit): ")
#         if user_input.lower() == "exit":
#             break
#         sio.emit("message", user_input)
#     sio.disconnect()


# import socketio
# import asyncio

# sio = socketio.AsyncClient(logger=True, engineio_logger=True)

# @sio.event
# async def connect():
#     print("Connected to the server")

# @sio.event
# def message(data):
#     print(f"Server says: {data}")
#     return "ok"

# @sio.event
# def disconnect():
#     print("Disconnected from the server")

# async def main():
#     await sio.connect('http://localhost:8000/', socketio_path="/socket.io")
#     # Register the response handler

#     while True:
#         user_input = input('Enter a message to send to the server (or type "exit" to quit): ')
#         if user_input == 'exit':
#             break
#         await sio.emit('message', user_input)
#     await sio.disconnect()


# if __name__ == '__main__':
#     asyncio.run(main())


import requests
import socketio
import asyncio
loop = asyncio.get_event_loop()



# r = requests.get("http://127.0.0.1:8000/")  # server prints "test"
sio = socketio.Client()
import logging
logging.basicConfig(level=logging.INFO)  # Set the logging level as needed


@sio.event
def transcription_response(data):
    print(f"client 1 {data}")
    # return "OK", 123



# async def main():
# sio.connect("http://127.0.0.1:8000/")  # server prints "on connect"
# while True:
#     user_input = input('Enter a message to send to the server (or type "exit" to quit):\n ')
#     if user_input == 'exit':
#         break
#     sio.emit('message', user_input)
#     sio.sleep(0.001)
# sio.disconnect()


import base64
import json
token = ""
try:
    sio.connect(url='ws://localhost:8000', socketio_path='/ws/socket.io', auth={"token": token})
except Exception as ex:
    print(ex)

# Read the audio file as binary data
with open("/Users/I1597/Downloads/5sec.wav", "rb") as audio_file:
    audio_data = audio_file.read()

# # Encode the audio data as base64
audio_base64 = base64.b64encode(audio_data).decode('utf-8')

# # Create a JSON object with the audio data
audio_json = {'audio_bytes': audio_base64}

# # Convert the JSON object to a string
json_str = json.dumps(audio_json)

data = {
    # "model_key": "46e73f4041ceda492b5ebeefb4823e2d",
    "request_id": "abcd1",
    "diarize": True,
    "sequence_id": 1,
    "audio_file": json_str,
    "max_num_speakers": 2,
    "min_num_speakers": 1,
    "audio_metadata": {
        "sample_width": 2,
        "sample_rate": 16000,
        "audio_format": "mp3",
        "channels": 1
    }
}
sio.emit('transcription_request', data)
sio.wait()
# sio.emit("message", "msg_1")  # prints client 1 msg_1
# sio.sleep(0.001)
# @sio.event
# async def error(data):
#     print(f"Received error: {data}")

# asyncio.run(main())