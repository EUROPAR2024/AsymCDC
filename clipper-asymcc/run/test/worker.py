from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from queue import Queue
from threading import Thread
from coder import Coder
import requests
import json
import time
import random
import socket
import numpy as np
import global_var
import torch
import torchvision.transforms as transforms
import io
from PIL import Image
import base64
import re
from util import in_dim, decode_in_dim
import pickle

TASK_ENCODE_TYPE = 0
TASK_INFER_TYPE = 1
TASK_REPAIR_TYPE = 2

transform = transforms.Compose([
    transforms.ToTensor()
])

transformPIL = transforms.Compose([
    transforms.ToPILImage()
])

def clipperRequest(addr, data):
    start = time.time()
    url = "http://{}:1337/pytorch-irevnet-app/predict".format(addr)
    # url = "http://localhost:1337/pytorch-irevnet-app/predict"
    req_json = json.dumps({
        'input': base64.b64encode(data).decode()
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    end = time.time()
    print("Inference time: {} ms".format(float(end - start) * 1000.0))
    # print(r.text)
    return r

def encodeTask(input, coder):
    print("=========encodeTask start=========")
    # bytes --> tensor
    id_list, encode_list = input
    encode_tensor_list = []
    for data in encode_list:
        encode_tensor_list.append(transform(Image.open(io.BytesIO(data))))
    final_tensor=torch.stack(encode_tensor_list,0)
    
    # encode
    encode_tensor = coder.encode(final_tensor)
    
    # tensor --> bytes
    imgByte = io.BytesIO()
    transformPIL(encode_tensor).save(imgByte, format = 'JPEG')
    encode_data = imgByte.getvalue()
    
    id_list.append(-1)
    encode_list.append(encode_data)
    
    infer_task = Task(TASK_INFER_TYPE, (id_list, encode_list))
    global_var.task_queue_put(infer_task)
    print("=========encodeTask end=========")

def inferTask(input, conf):
    print("=========inferTask start=========")
    id_list, data_list = input
    out_list = []
    outbij_list = []
    
    for i, data in enumerate(data_list):
        chosen = random.randint(0, conf.num_worker-1)
        chosen = 0
        chosen_ip = conf.cfg['worker_ips'][chosen]
        
        resp = clipperRequest(chosen_ip, data)

        json_output = eval(resp.json()["output"])
        tensor_out = pickle.loads(json_output[0])  #[1, 10]
        tensor_outbij = pickle.loads(json_output[1])  #[1, 512, 8, 8]
        
        out_list.append(tensor_out[0])
        outbij_list.append(tensor_outbij[0].reshape(decode_in_dim[conf.cfg['dataset']])) #[8, 64, 64]
        
    failed = -1
    if conf.cfg['sim_failure']:
        failed = random.randint(0, conf.cfg['ec_k'] - 1)  #[0,k-1]
        print("failed:",failed)
        
    for i, data in enumerate(out_list[:-1]):
        if failed >= 0 and i == failed:
            out_decode = torch.stack(out_list[:i] + out_list[i+1:], dim=0)  # k * [10] --> [k, 10]
            outbij_decode = torch.cat(outbij_list[:i] + outbij_list[i+1:], dim=0) # k * [8, 64, 64] --> [8*k, 64, 64]  
            repair_task = Task(TASK_REPAIR_TYPE, (id_list[i], out_decode, outbij_decode))
            global_var.task_queue_put(repair_task)
            continue
        global_var.resp_queue_put((id_list[i], data.numpy().argmax()))
    
    print("=========inferTask end=========")
    
def repairTask(input, coder):
    print("=========repairTask start=========")
    image_id, out_decode, outbij_decode = input
    
    # decode
    resp_tensor = coder.decode((out_decode, outbij_decode))
    
    global_var.resp_queue_put((image_id, resp_tensor.numpy().argmax()))
    print("=========repairTask end=========")
    return

class Task:
    def __init__(self, type, input) -> None:
        self.type = type
        self.input = input

class Worker:
    def __init__(self, conf, coder) -> None:
        self.conf = conf
        self.ec_k = self.conf.cfg['ec_k']
        self.coder = coder
        self.thrd_pool = ThreadPoolExecutor(max_workers=10)

    def process_task(self):
        while(True):
            task_list = []
            task = global_var.task_queue_get()
            if(task.type == TASK_ENCODE_TYPE):
                enc_task = self.thrd_pool.submit(encodeTask, task.input, self.coder)
                task_list.append(enc_task)
            elif(task.type == TASK_INFER_TYPE):
                inf_task = self.thrd_pool.submit(inferTask, task.input, self.conf)
                task_list.append(inf_task)
            elif(task.type == TASK_REPAIR_TYPE):
                rep_task = self.thrd_pool.submit(repairTask, task.input, self.coder)
                task_list.append(rep_task)
            wait(task_list, return_when=ALL_COMPLETED)
            
            
class Packager:
    def __init__(self, ec_k) -> None:
        self.ec_k = ec_k

    def pack_task(self):
        while(True):
            while global_var.origin_queue_qsize() < self.ec_k:
                continue
            id_list = []
            encode_group = []
            while len(id_list) < self.ec_k:
                id, data = global_var.origin_queue_get()
                id_list.append(id)
                encode_group.append(data)
            encode_task = Task(TASK_ENCODE_TYPE,(id_list,encode_group))
            global_var.task_queue_put(encode_task)


class Responsor:
    def __init__(self) -> None:
        pass

    def resp_task(self):
        while(True):
            if(global_var.resp_len > 0 and global_var.resp_queue_qsize() == global_var.resp_len):
                # merge result
                print("=====resp_task start=========")
                len = global_var.resp_len
                resp_str_list = []
                for i in range(len):
                    id,val = global_var.resp_queue_get()
                    resp_str_list.append(str(id))
                    resp_str_list.append(str(val))
                resp_str = ' '.join(resp_str_list)
            
                # send result to client
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("localhost",global_var.resp_port))
                sock.send(bytes(resp_str, encoding='utf-8'))
                sock.close()
                print("=====resp_task end=========")