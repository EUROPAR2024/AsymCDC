from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

def queue_init(length):
    # global origin_queue
    global task_queue
    global decode_queue
    global resp_queue
    global resp_port
    global resp_len
    global FLAG_STOP_THREAD
    # global clipper_req_pool
    
    # origin_queue = Queue(10000)
    task_queue = Queue(10000)
    resp_queue = Queue(10000)
    decode_queue = Queue(10000)
    resp_port = 50053
    resp_len = length
    FLAG_STOP_THREAD = False
    
    # clipper_req_pool = ThreadPoolExecutor(max_workers=20)


# def origin_queue_put(temp):
#     origin_queue.put(temp)

# def origin_queue_get():
#     return origin_queue.get()

# def origin_queue_qsize():
#     return origin_queue.qsize()

# def origin_queue_empty():
#     return origin_queue.empty()


def task_queue_put(temp):
    task_queue.put(temp)

def task_queue_get():
    return task_queue.get()

def task_queue_qsize():
    return task_queue.qsize()

def task_queue_empty():
    return task_queue.empty()



def decode_queue_put(temp):
    decode_queue.put(temp)

def decode_queue_get():
    return decode_queue.get()

def decode_queue_qsize():
    return decode_queue.qsize()

def decode_queue_empty():
    return decode_queue.empty()


def resp_queue_put(temp):
    resp_queue.put(temp)

def resp_queue_get():
    return resp_queue.get()

def resp_queue_qsize():
    return resp_queue.qsize()

def resp_queue_empty():
    return resp_queue.empty()


def task_finished():
    return resp_queue.qsize() == resp_len