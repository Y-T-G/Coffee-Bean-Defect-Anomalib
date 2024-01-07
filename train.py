# %% [markdown]
# # Coffee Bean Dataset

# %%
# !pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="API")
project = rf.workspace("yolo-annotated-dataset").project("usk-coffe")
dataset = project.version(2).download("darknet")


# %%
from glob import glob
import cv2
from pathlib import Path
import os

# %%
txts = glob("./USK-Coffe-2/test/*.txt")

# %%
!mkdir -p cropped/normal
!mkdir -p cropped/abnormal

# %%
# Crop images

for txt in txts:
  with open(txt, "r") as f:
    labels = f.readline().split(" ")
    labels = map(float, labels)
    label, xc, yc, w, h = labels
    txt_path = Path(txt)
    img_path = str(txt_path.parent / txt_path.stem) + ".jpg"
    class_label = "normal" if label != 0 else "abnormal"
    save_path = os.path.join("cropped", class_label, txt_path.stem + ".jpg")
    print(img_path)
    image = cv2.imread(img_path)
    image_width = image.shape[1]
    image_height = image.shape[0]
    x_min = (xc - w / 2) * image_width
    x_max = (xc + w / 2) * image_width
    y_min = (yc - h / 2) * image_height
    y_max = (yc + h / 2) * image_height

    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    cv2.imwrite(save_path, cropped_image)

# %% [markdown]
# # Train

# %%
import anomalib

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import get_experiment_logger

from pytorch_lightning import Trainer, seed_everything

import yaml
from pathlib import Path
from ast import literal_eval

# %%
model = "fastflow"
image_folder = "cropped"
batch_size = 32
val_ratio = 0.1

# %%
# model_config = yaml.safe_load(open(f"./src/anomalib/src/anomalib/models/{model}/config.yaml", "r"))
# config = model_config
config_path = (
    Path(f"{anomalib.__file__}").parent / f"models/{model}/config.yaml"
)
config = get_configurable_parameters(model_name=model, config_path=config_path)
config["dataset"] = yaml.safe_load(open("./config.yaml", "r"))
config["trainer"].update({"default_root_dir":"results/custom/run",
                          "max_epochs": 12})
config["project"].update({"path":"results/custom/run"})
config["optimization"].update({"export_mode":"torch"})

# del config["early_stopping"]

data_config = {
    "format": "folder",
    "name": str(Path(image_folder).name),
    "root": str(Path(image_folder)),
    "path": str(Path(image_folder)),
    "val_split_ratio": float(val_ratio),
    "train_batch_size": int(batch_size),
    "test_batch_size": int(batch_size),
}

config["dataset"].update(data_config)

if config.project.get("seed") is not None:
    seed_everything(config.project.seed)

yaml.dump(literal_eval(str(config)), open("config_dump.yaml","w"))

datamodule = get_datamodule(config)
model = get_model(config)
experiment_logger = get_experiment_logger(config)
callbacks = get_callbacks(config)

trainer = Trainer(
    **config.trainer, logger=experiment_logger, callbacks=callbacks
)

trainer.fit(model=model, datamodule=datamodule)

load_model_callback = LoadModelCallback(
    weights_path=trainer.checkpoint_callback.best_model_path
)
trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member


# %% [markdown]
# # Inference

# %%
from anomalib.deploy import TorchInferencer
import cv2
import matplotlib.pyplot as plt

# %%
image = cv2.imread("cropped/abnormal/1605_jpg.rf.6fb44c2809edbc36b2df72f80c98c382.jpg")[...,::-1]

# %%
inferencer = TorchInferencer(path="results/custom/run/weights/torch/model.pt")

# %%
predictions = inferencer.predict(image=image)

# %%
fig, axes = plt.subplots(1, 4)
fig.set_size_inches(10, 3)
axes[0].imshow(predictions.image)
axes[1].imshow(predictions.heat_map)
axes[2].imshow(predictions.anomaly_map)
axes[3].imshow(predictions.segmentations)

fig.savefig("plots.png")
