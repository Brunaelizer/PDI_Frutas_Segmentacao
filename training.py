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
def training():
    register_coco_instances("frutas_train", {}, "data/Teste/Train/test_coco.json", "")
    register_coco_instances("frutas_test", {}, "data/Teste/Test/test_coco.json", "")

    import random
    from detectron2.data import DatasetCatalog, MetadataCatalog

    dataset_dicts = DatasetCatalog.get("frutas_train")
    frutas_metadata = MetadataCatalog.get("frutas_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("frutas_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()