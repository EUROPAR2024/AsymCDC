from config import Config
import base64
import grpc
import infer_pb2_grpc, infer_pb2
from concurrent import futures
from queue import Queue
from threading import Thread
from coder import Coder
from worker import Worker,Packager,Responsor
import argparse
import global_var

class AgentService(infer_pb2_grpc.GrpcServiceServicer):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf

    def infer(self, request, context) -> infer_pb2.InferResponse:
        id = request.id
        input = request.input
        if input == b'':
            global_var.resp_len = id
        else:
            global_var.origin_queue_put((id,input))
        return infer_pb2.InferResponse(output = "Recived input!")

class Agent:
    def __init__(self, conf) -> None:
        self.conf = conf

        # server to response clients' request
        self.rpcserver = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    def serve(self) -> None:
        infer_pb2_grpc.add_GrpcServiceServicer_to_server(AgentService(self.conf), self.rpcserver)
        self.rpcserver.add_insecure_port('[::]:50052')
        self.rpcserver.start()
        self.rpcserver.wait_for_termination()
        
    def run(self) -> None:
        self.serve()

if __name__ == "__main__":
    global_var.queue_init()
    parser = argparse.ArgumentParser(description='i-NeDD clipper_deploy arguments')
    parser.add_argument("--conf", type=str, default="./config/simple.json", help="Path of the config file")
    args = parser.parse_args()
    
    conf = Config(args.conf)
    agent = Agent(conf)
    coder = Coder(conf)
    
    packager = Packager(conf.cfg['ec_k'])
    packThrd = Thread(target=packager.pack_task)
    packThrd.start()
    print("=========Packager strat==========")
    
    workerThrds = []
    for _ in range(conf.cfg['workerThrd_num']):
        worker = Worker(conf,coder)
        workerThrds.append(Thread(target=worker.process_task))
    
    for i, _ in enumerate(workerThrds):
        workerThrds[i].start()
        print("=========Worker start==========")
        
    responsor = Responsor()
    respThrd = Thread(target=responsor.resp_task)
    respThrd.start()
    print("=========Responsor strat==========")
    
    print("=========run start=========")
    agent.run()
    
    packThrd.join()
    respThrd.join()
    for thrd in workerThrds:
        thrd.join()
    