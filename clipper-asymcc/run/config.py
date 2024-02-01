import json

class Config:
    def __init__(self, path) -> None:
        with open(path, 'r') as infile:
            cfg = json.load(infile)
        
        self.cfg = cfg
        self.num_worker = len(cfg['worker_ips'])
        
        assert(cfg['fail_rate'] * cfg['ec_k'] <= 1)