import socketio
import asyncio
import base64
import json
import time

sio = socketio.AsyncClient()

async def handle_response(data):
    print(f"Received response: {data}")

@sio.event
async def message_1(data):
    print(f"Received response: {data}")

async def main():
    with open("/Users/I1597/Downloads/5sec.wav", "rb") as audio_file:
        audio_data = audio_file.read()

# Encode the audio data as base64
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    # Create a JSON object with the audio data
    audio_json = {'audio_bytes': audio_base64}

    # Convert the JSON object to a string
    json_str = json.dumps(audio_json)

    data = {
        "unique_key": "abcd",
        "diarize": True,
        "audio_file": json_str
    }

    await sio.connect('ws://localhost:8000/transcribe', socketio_path='/ws/socket.io')
    await sio.emit('message', data)
    await sio.wait()
    with open("/Users/I1597/Downloads/5sec.wav", "rb") as audio_file:
        audio_data = audio_file.read()

# Encode the audio data as base64
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    # Create a JSON object with the audio data
    audio_json = {'audio_bytes': audio_base64}

    # Convert the JSON object to a string
    json_str = json.dumps(audio_json)

    data = {
        # "model_key": "46e73f4041ceda492b5ebeefb4823e2d",
        "unique_key": "abcdef",
        "diarize": True,
        "audio_file": json_str
    }

    await sio.emit('message', data)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(main())  # Start the Socket.IO communication in the background
    try:
        loop.run_forever()  # Continue running the event loop indefinitely
    except KeyboardInterrupt:
        pass  # Exit gracefully on Ctrl+C
