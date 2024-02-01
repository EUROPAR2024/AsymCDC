import torch
import json
import numpy as np
import time
from threading import Thread
import json
import base64
import requests
from datetime import datetime
import time
from PIL import Image
from IPython.display import display
import numpy
import re
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


def inferTask():
    headers = {'Content-type': 'application/json'}
    url = "http://localhost:1337/pytorch-irevnet-app/predict"
    start = time.time()
    req_json = json.dumps({ "input": base64.b64encode(open('../datatest/4.jpg', "rb").read()).decode() })
    r = requests.post(url, headers=headers, data=req_json)
    end = time.time()
    print("post: {} ms".format(float(end-start)*1000.0))

    str_output = r.json()["output"]
    json_output = eval(str_output)
    start = time.time()
    np_out = np.fromstring(json_output[0], sep=' ')
    np_outbij = np.fromstring(json_output[1], sep=' ').reshape((512,8,8))
    end = time.time()
    print("fromstring: {} ms".format(float(end-start)*1000.0))

class Worker:
    def __init__(self) -> None:
        self.thrd_pool = ThreadPoolExecutor(max_workers=10)

    def process_task(self):
        while(True):
            time.sleep(3)
            print("process_task")
            task_list = []
            inf_task = self.thrd_pool.submit(inferTask)
            task_list.append(inf_task)
            wait(task_list, return_when=ALL_COMPLETED)

workerThrds = []
for _ in range(4):
    worker = Worker()
    x = Thread(target=worker.process_task)
    x.start()
    workerThrds.append(x)
    print("=========Worker start==========")
for thrd in workerThrds:
    thrd.join()