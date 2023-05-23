from typing import Dict, Any, Optional
from os import path
from torch.utils.tensorboard import SummaryWriter


def get_reporter(name: str, desc: Optional[str] = None):
    writer = SummaryWriter(comment='_' + name)
    times_counter: Dict[str, int] = dict()

    if desc is not None:
        assert writer.log_dir is not None
        with open(path.join(writer.log_dir, 'desc.json'), 'w') as f:
            f.write(desc)

    def reporter(info: Dict[str, Any]):
        nonlocal times_counter

        for k, v in info.items():
            if not k in times_counter:
                times_counter[k] = 0
            writer.add_scalar(k, v, times_counter[k])
            times_counter[k] = times_counter[k] + 1

    return reporter
