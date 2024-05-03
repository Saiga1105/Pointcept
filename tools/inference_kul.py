"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch

####
from pointcept.datasets import build_dataset, collate_fn
import pointcept.utils.comm as comm
import torch
import os
from pointcept.engines.defaults import create_ddp_model
from collections import OrderedDict
import time
import numpy as np
from pointcept.utils.misc import make_dirs
import torch.nn.functional as F
from pointcept.models import build_model


def collate_fn(batch):
    return batch

def build_inference_model(cfg):
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = create_ddp_model(
        model.cuda(),
        broadcast_buffers=False,
        find_unused_parameters=cfg.find_unused_parameters,
    )
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(cfg.weight)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value
        model.load_state_dict(weight, strict=True)

    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(cfg.weight))
    return model

def main_worker(cfg):
    # cfg = default_setup(cfg)
    # tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    
    # print(f'cfg.batch_size_test_per_gpu:{cfg.batch_size_test_per_gpu}')
    # tester.test()
    cfg = default_setup(cfg)
    print('building dataset')
    test_dataset = build_dataset(cfg.data.test)

    # print(f'world size :{comm.get_world_size()}')
    if comm.get_world_size() > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
        
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    
    model = build_inference_model(cfg)
    model.eval()

    save_path = os.path.join(cfg.data.test.data_root, "test")
    make_dirs(save_path)
    
    print('Compiling results')
    for idx, data_dict in enumerate(test_loader):
        
        data_dict = data_dict[0]  # Assuming batch size is 1 for simplicity
        segment = data_dict.pop("segment")
        data_name = data_dict.pop("name")
        
        print(f"Processing {data_name} ({idx + 1}/{len(test_loader)})")
        
        pred_save_path = os.path.join(save_path, f"{data_name}.npy")

        pred = torch.zeros((segment.size, cfg.data.num_classes))  # CPU tensor otherwise we have an aggregation on GPU
        for fragment in data_dict['fragment_list']:
            input_dict = collate_fn([fragment])[0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            idx_part = input_dict.pop("index")
            with torch.no_grad():
                pred_part = model(input_dict)["seg_logits"]
                pred_part = F.softmax(pred_part, -1).cpu()  # Move to CPU immediately
                for i in range(pred_part.shape[0]):
                    pred[idx_part[i], :] += pred_part[i]

        pred = pred.argmax(dim=1).numpy()
        np.save(pred_save_path, pred)

    print("DONE.")


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    # print(f'number of gpu per machine: {args.num_gpus}')
    # print(f'number of machines: {args.num_machines}')
    # print(f'machine rank: {args.machine_rank}')
    
    # cfg['num_gpus'] = 3
    
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()

#export PYTHONPATH=./
#python tools/inference_mb.py --config-file '/home/mbassier/code/Scan-to-BIM-CVPR-2024/data/t1/config.py' --num-gpus 3 --num-machines 1 --machine-rank 0