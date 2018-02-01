import json
import asyncio
import websockets
import json
import sys, os
sys.path.append('/home/boat/openface/')
from websocket.small_classifier import TestClassifier


async def train():
    pass


async def recognize(websocket, path):
    print('connection begin...')
    json_data = await websocket.recv()
    print("< {}".format(json_data))
    greeting = "receive complete"
    data = json.loads(json_data)
    if data['type'] == "recognize":
        result = model.infer(le, clf, data['path'], data['multi_face'])
    else:
        result = train()
    await websocket.send(json.dumps(result))
    print(data['multi_face'])
    if data['multi_face']:
        print("true")
    print("> {}".format(greeting))

start_server = websockets.serve(recognize, 'localhost', 8050, max_size=2 ** 23)
model = TestClassifier(True)
le, clf = model.load_model("../generated-embeddings/classifier.pkl")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

# async def test_predict():
#     model = TestClassifier(True)
#     model.infer("../generated-embeddings/classifier.pkl", "../aaron.jpg")
#     pass


# asyncio.get_event_loop().run_until_complete(test_predict())
# asyncio.get_event_loop().run_forever()