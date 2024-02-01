from config import Config
from concurrent import futures
from threading import Thread
from coder import Coder
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from queue import Queue
import queue
import requests
import json
import time
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import io
from PIL import Image
import base64
import pickle
from util import in_dim, decode_in_dim
import global_var


TASK_ENCODE_TYPE = 0
TASK_INFER_TYPE = 1
TIME_OUT = 5
FLAG_STOP_THREAD = False
# TASK_REPAIR_TYPE = 2
# TASK_RESPOND_TYPE = 3


transform = transforms.Compose([
    transforms.ToTensor()
])

transformPIL = transforms.Compose([
    transforms.ToPILImage()
])

class ImageObject:
    def __init__(self, id, fpath):
        self.id = id
        self.path = fpath
        self.byte = open(fpath, "rb").read()

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
    INFO.add_time(float(end - start) * 1000.0)
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
    start = time.time()
    encode_tensor = coder.encode(final_tensor)
    end = time.time()
    print("encode costs: {} ms".format(float(end-start)*1000.0))
    
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
    start = time.time()
    id_list, data_list = input
    out_list = []
    outbij_list = []
    
    # req_Thrds = []
    for i, data in enumerate(data_list):
        chosen = random.randint(0, conf.num_worker-1)
        # chosen = 0
        chosen_ip = conf.cfg['worker_ips'][chosen]
        resp = clipperRequest(chosen_ip, data)
        # req_Thrds.append(global_var.clipper_req_pool.submit(clipperRequest, chosen_ip, data))
    
    # for i, _ in enumerate(data_list):
        # resp = req_Thrds[i].result()
        json_output = eval(resp.json()["output"])
        tensor_out = pickle.loads(json_output[0])  #[1, 10]
        tensor_outbij = pickle.loads(json_output[1])  #[1, 512, 8, 8]
        
        out_list.append(tensor_out[0])
        outbij_list.append(tensor_outbij[0].reshape(decode_in_dim[conf.cfg['dataset']])) #[8, 64, 64]
    
    end = time.time()
    print("inferTask total costs: {} ms".format(float(end-start)*1000.0))
    
    failed = -1
    if conf.cfg['sim_failure']:
        failed = random.randint(0, conf.cfg['ec_k'] - 1)  #[0,k-1]
        failed = 0
        print("failed:",failed)
        zero_tensor = torch.zeros_like(out_list[failed])
        out_list[failed] = zero_tensor
        
        out_decode = torch.stack(out_list, dim=0)  # (k+1) * [10] --> [k, 10]
        outbij_decode = torch.cat(outbij_list[:failed] + outbij_list[failed+1:], dim=0) # k * [8, 64, 64] --> [8*k, 64, 64]  
        # repair_task = Task(TASK_REPAIR_TYPE, (id_list[failed], out_decode, outbij_decode))
        # global_var.task_queue_put(repair_task)
        global_var.decode_queue_put((id_list[failed], out_decode, outbij_decode))
        
    for i, data in enumerate(out_list[:-1]):
        if failed < 0 or i != failed:
            global_var.resp_queue_put((id_list[i], data.numpy().argmax()))
            print(id_list[i])
    
    print("=========inferTask end=========")

def respondTask(outpath):
    print("=====resp_task start=========")
    len = global_var.resp_len
    resp_str = ""
    for i in range(len):
        id,val = global_var.resp_queue_get()
        resp_str += str(id)+' '+str(val)+ '\n'
    print(resp_str)
    with open(outpath, 'w') as f:
        f.write(resp_str)
    print("=====resp_task end=========")

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
            try:
                task = global_var.task_queue.get(block=True, timeout=TIME_OUT)
            except queue.Empty:
                if FLAG_STOP_THREAD:
                    print("Worker end")
                    break
                else:
                    continue
            
            if(task.type == TASK_ENCODE_TYPE):
                enc_task = self.thrd_pool.submit(encodeTask, task.input, self.coder)
                task_list.append(enc_task)
            elif(task.type == TASK_INFER_TYPE):
                inf_task = self.thrd_pool.submit(inferTask, task.input, self.conf)
                task_list.append(inf_task)
            # elif(task.type == TASK_REPAIR_TYPE):
            #     rep_task = self.thrd_pool.submit(repairTask, task.input, self.coder)
            #     task_list.append(rep_task)
            # elif(task.type == TASK_RESPOND_TYPE):
            #     resp_task = self.thrd_pool.submit(respondTask, self.out)
            #     task_list.append(resp_task)
            wait(task_list, return_when=ALL_COMPLETED)

class Repairer:
    def __init__(self, ec_k, coder) -> None:
        self.ec_k = ec_k
        self.coder = coder

    def decode_task(self):
        while(True):
            try:
                input = global_var.decode_queue.get(block=True, timeout=TIME_OUT)
            except queue.Empty:
                if FLAG_STOP_THREAD:
                    print("Repairer end")
                    break
                else:
                    continue
            image_id, out_decode, outbij_decode = input

            print("=========repairTask start=========")
            # decode
            start = time.time()
            resp_tensor = self.coder.decode((out_decode, outbij_decode))
            end = time.time()
            print("decode costs: {} ms".format(float(end-start)*1000.0))
    
            global_var.resp_queue_put((image_id, resp_tensor.numpy().argmax()))
            print("=========repairTask end=========")

class Info:
    def __init__(self) -> None:
        self.inferTime = []
    
    def add_time(self,t):
        self.inferTime.append(t)
    
    def output(self):
        # assert(len(self.inferTime) == self.questNum)
        print(len(self.inferTime))
        print(np.median(self.inferTime))

INFO = Info()


def SendTask(path_list, k, time_send_group):
    for i in range(global_var.resp_len // k):
        id_list = []
        encode_group = []
        for j,fpath in enumerate(path_list[i*k : i*k + k]):
            img = ImageObject(i*k + j, fpath)
            id_list.append(img.id)
            encode_group.append(img.byte)
            time.sleep(time_send_group)
        encode_task = Task(TASK_ENCODE_TYPE,(id_list,encode_group))
        global_var.task_queue.put(encode_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='i-NeDD clipper_deploy arguments')
    parser.add_argument("--conf", type=str, default="/root/i-NeDD/clipper-inedd/run/config/simple.json", help="Path of the config file")
    parser.add_argument("--path", type=str, default="/root/i-NeDD/clipper-inedd/run/datatest/cifar10/test", help="Path of the input file")
    
    args = parser.parse_args()
    conf = Config(args.conf)
    coder = Coder(conf)
    path_list = [os.path.join(args.path, file) for file in os.listdir(args.path)]
    global_var.queue_init()
    global_var.resp_len = len(path_list[:100])
    
    workerThrds = []
    for _ in range(conf.cfg['workerThrd_num']):
        worker = Worker(conf,coder)
        workerThrds.append(Thread(target=worker.process_task))
    for i, _ in enumerate(workerThrds):
        print("=========Worker start==========")
        workerThrds[i].start()
        
    repairer = Repairer(conf, coder)
    repairThrd = Thread(target=repairer.decode_task)
    print("=========Repairer start==========")
    repairThrd.start()
            
    SendTask(path_list, conf.cfg['ec_k'], 1/(conf.cfg['query_rate']/conf.cfg['ec_k']))
    
    while(True):
        if global_var.resp_queue_qsize() == global_var.resp_len:
            FLAG_STOP_THREAD = True
            break
        time.sleep(TIME_OUT)
    respondTask(conf.cfg['output_path'])
    INFO.output()