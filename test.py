import cv2
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
import random
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import matplotlib.pyplot as plt
import matplotlib

from detectron2.utils.visualizer import ColorMode

def test():
    register_coco_instances("frutas_train", {}, "data/Teste/Train/test_coco.json", "")
    register_coco_instances("frutas_test", {}, "data/Teste/Test/test_coco.json", "")

    from detectron2.data import DatasetCatalog, MetadataCatalog

    dataset_dicts = DatasetCatalog.get("frutas_train")
    frutas_metadata = MetadataCatalog.get("frutas_train")

    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("frutas_test", )
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = DatasetCatalog.get("frutas_test")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread("data/Teste/Test/berinjela.png")
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=frutas_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        matplotlib.use('tkagg')
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()


