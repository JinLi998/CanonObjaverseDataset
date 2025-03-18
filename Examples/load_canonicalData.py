"""Obtain a canonical dataset
"""

import argparse

from Src.DataDownload.Download import DownloadGLBs
from Src.Glb2Obj.glb2Obj import Relay2Obj
from Src.Obj2Canonicalization.canon_obj import canon_meshs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and canonicalize the dataset')
    parser.add_argument('--save_dir', type=str, default='results/dataset', help='Directory to save the dataset')
    parser.add_argument('--data_name', type=str, default='test', help='data_name')  # 'canon' 'test'
    args = parser.parse_args()

    save_dir=args.save_dir
    data_name=args.data_name
    DownloadGLBs(data_name=data_name, save_dir=save_dir)  # 'canon' 
    Relay2Obj(save_dir)
    canon_meshs(save_dir)

