import os
import shutil
from glob import glob

# CHECKPOINT_ROOT = '/apdcephfs/private_tobinwu/Research/snapshots/Mask2Former'
CHECKPOINT_ROOT = '/youtu/xlab-team2-2/persons/tobinwu/Research/snapshots/Mask2Former'


def clear():
    last_info_files = glob(f"{CHECKPOINT_ROOT}/**/last_checkpoint", recursive=True)
    for info in last_info_files:
        if not os.path.isfile(info):
            continue
        with open(info, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1
        last_ckpt = lines[0]

        dirpath = os.path.dirname(info)
        ckpt_paths = glob(f"{dirpath}/*.pth")
        for path in ckpt_paths:
            if os.path.basename(path) == last_ckpt:
                print('continue:', last_ckpt)
                continue
            os.remove(path)
        if os.path.isdir(f"{dirpath}/inference"):
            shutil.rmtree(f"{dirpath}/inference")


if __name__ == '__main__':
    clear()
