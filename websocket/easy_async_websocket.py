import json
import asyncio
import websockets
import json
import sys, os, multiprocessing

sys.path.append('/home/boat/openface/')
from websocket.small_classifier import TestClassifier
from websocket.small_align_dlib import TestAlign
import subprocess
import concurrent.futures

align_path = "/home/boat/webserver_for_socket/face_api/recognition/static/recognition/align-my-photo"
image_path = "/home/boat/webserver_for_socket/face_api/recognition/static/recognition/my_photo"


async def train(id_classifier):
    global status
    status = 'aligned images'
    await asyncio.sleep(1)
    model_align = TestAlign(True, align_path, landmarkMap[0])
    model_align.alignMain(image_path)
    status = 'generated embeddings'
    await asyncio.sleep(1)
    result = subprocess.check_output(['th'
                                         , './batch-represent/main.lua'
                                         , '-outDir'
                                         , '../generated-embeddings/'
                                         , '-data'
                                         , align_path])
    status = "training"
    await asyncio.sleep(1)
    model_recognize.train(clf, '../generated-embeddings/', id_classifier)
    status = "complete"
    # http://websockets.readthedocs.io/en/stable/intro.html see both topic and registration topic
    await asyncio.sleep(2)
    return result


async def recognize(websocket, path):
    print("cpu num" + str(multiprocessing.cpu_count()))
    print('connection begin...')
    json_data = await websocket.recv()
    print("< {}".format(json_data))
    greeting = "receive complete"
    data = json.loads(json_data)
    if status == "complete":
        if data['type'] == "recognize":
            result = model_recognize.infer(le, clf, data['path'], data['multi_face'])
            await websocket.send(json.dumps(result))
        elif data['type'] == "train":
            result = await train(data['id'])
            await websocket.send(result.decode("utf-8", "ignore"))
        else:
            await websocket.send(json.dumps({"status": status}))

    else:
        await websocket.send(json.dumps({"status": status}))
    print("> {}".format(greeting))


landmarkMap = [
    'outerEyesAndNose',
    'innerEyesAndBottomLip'
]
classifier = [
                 'LinearSvm',
                 'GridSearchSvm',
                 'GMM',
                 'RadialSvm',
                 'DecisionTree',
                 'GaussianNB',
                 'DBN'],
status = str("complete")
model_recognize = TestClassifier(True)
le, clf = model_recognize.load_model("../generated-embeddings/classifier.pkl")

start_server = websockets.serve(recognize, 'localhost', 8050, max_size=2 ** 23)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

# with concurrent.futures.ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:

