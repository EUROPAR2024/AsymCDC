from config import Config
import base64
import grpc
import infer_pb2, infer_pb2_grpc
import time
from datetime import datetime
import socket
import argparse
import os

bufsize = 1024 * 1024

def send_request(id, input, port, identity) -> None:
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = infer_pb2_grpc.GrpcServiceStub(channel=channel)
        response = stub.infer(infer_pb2.InferRequest(id=id, input=input, port=port, identity=identity))
    print("Client received: " + response.output)


class Image:
    def __init__(self, id, fpath):
        self.id = id
        self.path = fpath
        self.byte = open(fpath, "rb").read()
        self.timestamp = fpath + '-' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')

class Client:
    def __init__(self, conf, port) -> None:
        self.conf = conf
        self.port = port

    def infer(self, fpath_list):
        start_t = time.time()
        
        for i,fpath in enumerate(fpath_list):
            img = Image(i, fpath)
            send_request(id=img.id, input=img.byte, port=self.port, identity=img.timestamp)

        send_request(id=len(fpath_list), input=b'', port=self.port, identity='')
        # wait for inference result
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost",self.port))
        sock.listen(1)
        conn, _ = sock.accept()
        data = conn.recv(bufsize)
        sock.close()

        print("Get inference result: {}".format(str(data, encoding='utf-8')))

        end_t = time.time()

        total_time = float(end_t - start_t)*1000.0

        print("Inference task costs: {} ms".format(total_time))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='i-NeDD client arguments')
    parser.add_argument("--path", type=str, default="./datatest", help="Path of the input file")
    parser.add_argument("--conf", type=str, default="./config/simple.json", help="Path of the config file")
    parser.add_argument("--port", type=int, default=50053, help="Listening port to recv")
    
    args = parser.parse_args()
    conf = Config(args.conf)
    
    client = Client(conf,args.port)
    path_list = [os.path.join(args.path, file) for file in os.listdir(args.path)]
    client.infer(path_list[:2])