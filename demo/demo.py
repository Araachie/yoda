import os
import wget
from glob import glob

from model import Model
from lutils.configuration import Configuration

from PIL import Image
import torch
from torchvision import transforms as T

import matplotlib.pyplot as plt
import numpy as np

from functools import partial


def load_model(dataset_name: str):
    config_path = f"configs/{dataset_name}.yaml"
    dst = f"model_weights/{dataset_name}/model.pth"
    dst_ae = f"model_weights/{dataset_name}/ae"
    dst_ae += ".pth" if dataset_name.startswith("clevrer") else ".ckpt"

    im_res = 256 if dataset_name == "bair" else 128
    model_url = f"https://huggingface.co/cvg-unibe/yoda_{dataset_name}_{im_res}/resolve/main/model.pth"
    ae_url = f"https://huggingface.co/cvg-unibe/yoda_{dataset_name}_{im_res}/resolve/main/vqvae"
    ae_url += ".pth" if dataset_name.startswith("clevrer") else ".ckpt"

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        wget.download(model_url, out=dst)
    if not os.path.exists(dst_ae):
        wget.download(ae_url, out=dst_ae)

    config = Configuration(config_path)
    config["model"]["autoencoder"]["ckpt_path"] = dst_ae
    model = Model(config["model"])
    model.load_from_ckpt(dst)
    model.eval()

    print("Loaded the model from", dst)

    return model, im_res


class Demo:
    def __init__(self, dataset_name: str, device: str = "cpu", steps: int = 20):
        self.dataset_name = dataset_name
        self.model, self.im_res = load_model(dataset_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.steps = steps

        self.image_filenames = glob(os.path.join("media", dataset_name, "*.png"))
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.reinit()

    def run(self, index: int = None):
        if index is None:
            index = np.random.randint(len(self.image_filenames))
        dimage = Image.open(self.image_filenames[index]).convert("RGB").resize((self.im_res, self.im_res))
        dimage = self.transforms(dimage)
        self.images.append(dimage)

        image = (np.clip(dimage.permute(1, 2, 0), -1, 1) + 1) / 2

        fig = plt.figure(figsize=(9.5, 4))
        fig.suptitle(f'Interactive video generation on {self.dataset_name}, idx: {index}', fontsize=16)
        self.ax = plt.subplot(1, 1, 1)

        self.ax.imshow(image)

        fig.canvas.mpl_connect('button_press_event', self.onclick)
        fig.canvas.mpl_connect('button_release_event', self.onrelease)
        fig.canvas.mpl_connect('key_press_event', partial(self.onkey))

    def reinit(self):
        self.x = [[]]
        self.y = [[]]
        self.new_vec = [True]
        self.images = []
        self.ax = None

    @staticmethod
    def set_flow(flow, h, w, vec):
        flow[0, 0, h, w] = vec[0]
        flow[0, 1, h, w] = vec[1]
        flow[0, 2, h, w] = 1.0

    def onclick(self, event):
        assert self.new_vec[-1]

        self.x[-1].append([int(event.xdata)])
        self.y[-1].append([int(event.ydata)])
        self.new_vec.append(False)

    def onrelease(self, event):
        assert not self.new_vec[-1]

        self.x[-1][-1].append(int(event.xdata))
        self.y[-1][-1].append(int(event.ydata))
        self.new_vec.append(True)

        vx = self.x[-1][-1][1] - self.x[-1][-1][0]
        vy = self.y[-1][-1][1] - self.y[-1][-1][0]
        assert self.ax is not None
        self.ax.arrow(self.x[-1][-1][0], self.y[-1][-1][0], vx, vy, width=2, color="r")

    @torch.no_grad()
    def onkey(self, event):
        flows = torch.zeros([1, 1, 3, self.im_res, self.im_res]).to(self.device)
        if len(self.x[-1]) == 0:
            flows = None
        else:
            for i in range(len(self.x[-1])):
                Demo.set_flow(
                    flows[:, 0],
                    self.y[-1][i][0],
                    self.x[-1][i][0],
                    (
                        self.x[-1][i][1] - self.x[-1][i][0],
                        self.y[-1][i][1] - self.y[-1][i][0]
                    )
                )

        generated_frames = self.model.generate_frames(
            torch.stack(self.images, dim=0).to(self.device).unsqueeze(0),
            flows,
            steps=self.steps,
            num_frames=1,
            past_horizon=-1,
            skip_past=False,
            verbose=False)
        self.images.append(generated_frames[0, -1].cpu())

        self.ax.clear()
        self.ax.imshow((self.images[-1].permute(1, 2, 0).numpy() + 1) / 2)
        self.ax.annotate(f"{len(self.x)}", (self.im_res - 20, 20), fontsize=18)

        self.x.append([])
        self.y.append([])
