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

r = requests.get("http://127.0.0.1:5000/test") # server prints "test"
cl = socketio.Client()
cl2 = socketio.Client()


@cl.on("event_name")
def foo(data):
    print(f"client 1 {data}")


@cl2.on("event_name")
def foo2(data):
    print(f"client 2 {data}")


cl.connect("http://127.0.0.1:5000/") # server prints "on connect"
cl2.connect("http://127.0.0.1:5000/")
cl.emit("direct", "msg_1") # prints client 1 msg_1
cl2.emit("broadcast", "msg_2") # prints client 2 msg_2 and client 1 msg_2