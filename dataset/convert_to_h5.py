import argparse
import os
import sys
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from h5 import HDF5Maker


def get_seq(video_dirs, extension="png", image_size=64, with_flows=False):
    for f in video_dirs:
        images = sorted(glob(os.path.join(f, f"*.{extension}")))
        image_seq = []

        for image_name in images:
            img = Image.open(image_name).convert("RGB").resize((image_size, image_size))
            arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
            image_seq.append(arr)

        if with_flows:
            flows = [os.path.join("/", os.path.join(*(x.split("/")[:-1])), "flows",
                                  ".".join([x.split("/")[-1].split(".")[0], "npy"])) for x in images]
            flow_seq = []
            for flow_name in flows:
                arr = np.load(flow_name)
                flow_seq.append(arr)

            yield image_seq, flow_seq
        else:
            yield image_seq


def make_h5(
        data_root,
        split,
        image_size=64,
        extension="png",
        with_flows=False,
        out_dir='./h5_ds',
        vids_per_shard=100000,
        force_h5=False):

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True, flow=with_flows)

    data_dir = os.path.join(data_root, split)
    video_dirs = [os.path.join(data_dir, vd) for vd in sorted(os.listdir(data_dir))]

    seq_generator = get_seq(video_dirs, extension=extension, image_size=image_size, with_flows=with_flows)

    for seq in tqdm(seq_generator, total=len(video_dirs)):
        try:
            if not with_flows:
                h5_maker.add_data(seq, dtype='uint8')
            else:
                h5_maker.add_data(seq[0], dtype='uint8', flow=seq[1], flowtype='float32')
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            print("Ctrl+C!!")
            break
        except:
            e = sys.exc_info()[0]
            print("ERROR:", e)

    h5_maker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--data_dir', type=str, help="Directory with videos")
    parser.add_argument('--image_size', type=int, help="Resolution to resize the images to")
    parser.add_argument('--extension', type=str, help="Video frames extension")
    parser.add_argument('--with_flows', action="store_true", help="If defined, add flows to the dataset")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        make_h5(
            data_root=args.data_dir,
            split=split,
            extension=args.extension,
            with_flows=args.with_flows,
            image_size=args.image_size,
            out_dir=os.path.join(args.out_dir, 'train'))
