# solution.py

import asyncio
import websockets
import json

# Define the VideoCollabEditor class
class VideoCollabEditor:
    def __init__(self):
        self.clients = set()
        self.video_data = {}

    async def register(self, websocket):
        self.clients.add(websocket)
        await self.notify_clients()

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        await self.notify_clients()

    async def notify_clients(self):
        message = json.dumps(self.video_data)
        await asyncio.wait([client.send(message) for client in self.clients])

    async def handle_message(self, websocket, message):
        data = json.loads(message)
        if data['action'] == 'edit':
            self.video_data.update(data['changes'])
            await self.notify_clients()

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            await self.unregister(websocket)

# Start the WebSocket server
async def main():
    editor = VideoCollabEditor()
    async with websockets.serve(editor.handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())