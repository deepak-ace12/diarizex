# import socketio
# import eventlet

# sio = socketio.Server(cors_allowed_origins="*")
# app = socketio.WSGIApp(sio)

# @sio.on("connect")
# def connect(sid, environ):
#     print(f"Client {sid} connected")

# @sio.on("message")
# def message(sid, data):
#     print(f"Client {sid} says: {data}")
#     sio.emit("message", f"Server received: {data}", room=sid)

# @sio.on("disconnect")
# def disconnect(sid):
#     print(f"Client {sid} disconnected")

# if __name__ == "__main__":
#     eventlet.wsgi.server(eventlet.listen(("localhost", 5000)), app)


from fastapi import FastAPI
from socket_manager import SocketManager

app = FastAPI()
sio = SocketManager(app=app)

@app.get('/')
async def root():
    print("WORKS")
    return {'message': 'Hello world'}

@app.sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")

@app.sio.on("message")
async def message_handler(sid, data):
    print(f"Client {sid} says: {data}")
    await app.sio.emit("message_1", f"Server received: {data}", room=sid)

@app.sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")


if __name__ == '__main__':
    import logging
    import sys

    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stdout)

    import uvicorn

    uvicorn.run("server:app", host='0.0.0.0', port=5000, reload=True)



# from typing import Any

# import uvicorn
# from fastapi import FastAPI

# import socketio

# sio: Any = socketio.AsyncServer(async_mode="asgi")
# socket_app = socketio.ASGIApp(sio)
# app = FastAPI()


# @app.get("/test")
# async def test():
#     print("test")
#     return "WORKS"


# app.mount("/", socket_app)  # Here we mount socket app to main fastapi app


# @sio.on("connect")
# async def connect(sid, env):
#     print("on connect", sid)


# @sio.on("direct")
# async def direct(sid, msg):
#     print(f"direct {msg} -{sid}")
#     await sio.emit("event_name1", msg, room=sid)  # we can send message to specific sid


# @sio.on("broadcast")
# async def broadcast(sid, msg):
#     print(f"broadcast {msg}")
#     await sio.emit("event_name2", msg, room=sid)  # or send to everyone


# @sio.on("disconnect")
# async def disconnect(sid):
#     print("on disconnect")

# @sio.event
# def connect_error(data):
#     print(f"Connection error: {data}")

# if __name__ == "__main__":
#     kwargs = {"host": "0.0.0.0", "port": 5000}
#     kwargs.update({"reload": True})
#     uvicorn.run("server:app", **kwargs)
