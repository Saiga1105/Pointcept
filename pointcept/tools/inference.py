from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
from pointcept.datasets import build_dataset, collate_fn
import pointcept.utils.comm as comm
import torch
from pointcept.engines.defaults import create_ddp_model
import os
from collections import OrderedDict
import time
import numpy as np
from pointcept.utils.misc import make_dirs
import torch.nn.functional as F
from pointcept.models import build_model

torch.multiprocessing.set_sharing_strategy('file_system')

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
        # self.logger.info(f"Loading weight at: {self.cfg.weight}")
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
        # self.logger.info(
        #     "=> Loaded weight '{}' (epoch {})".format(
        #         self.cfg.weight, checkpoint["epoch"]
        #     )
        # )
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(cfg.weight))
    return model
    

def main_worker(cfg):
    cfg = default_setup(cfg)
    test_dataset = build_dataset(cfg.data.test)

    if comm.get_world_size() > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size_test_per_gpu,
        shuffle=False,
        num_workers=cfg.batch_size_test_per_gpu,
        pin_memory=True,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    
    # logger = get_root_logger()
    # logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    model = build_inference_model(cfg)
    model.eval()

    save_path = os.path.join(cfg.save_path, "result")
    make_dirs(save_path)
    
    for idx, data_dict in enumerate(test_loader):
        data_dict = data_dict[0]  # current assume batch size is 1
        fragment_list = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")
        data_name = data_dict.pop("name")
        pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))

        pred = torch.zeros((segment.size, cfg.data.num_classes)).cuda()
        for i in range(len(fragment_list)):
            fragment_batch_size = 1
            s_i, e_i = i * fragment_batch_size, min(
                (i + 1) * fragment_batch_size, len(fragment_list)
            )
            input_dict = collate_fn(fragment_list[s_i:e_i])[0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            idx_part = input_dict["index"]            
            with torch.no_grad():
                pred_part = model(input_dict)["seg_logits"]  # (n, k)
                pred_part = F.softmax(pred_part, -1)
                if cfg.empty_cache:
                    torch.cuda.empty_cache()
                bs = 0                
                for be in input_dict["offset"]:
                    pred[idx_part[bs:be], :] += pred_part[bs:be]
                    bs = be        
        pred = pred.max(1)[1].data.cpu().numpy()
        np.save(pred_save_path, pred)

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    launch(
        main_worker,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url='auto',
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
