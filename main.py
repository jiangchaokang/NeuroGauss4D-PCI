import os
import sys
import torch
import numpy as np
import random
from config.config import npci_config
from model.Runner import Runner


def set_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    # fix the seed
    set_seed(0)

    pid = os.getpid()
    print("PID:{}".format(pid))

    args = npci_config()
    print(args)

    runner = Runner(args)

    if not args.demo:
        if args.SceneFlow:
            runner.Accumul_sf()
        else:
            runner.loop()
    else:
        runner.demo()
        # runner.visualization()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

