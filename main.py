import cv2
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
import matplotlib




register_coco_instances("frutas_train", {}, "data/Teste/Train/test_coco.json", "")
register_coco_instances("frutas_test", {}, "data/Teste/Test/test_coco.json", "")

import random
from detectron2.data import DatasetCatalog, MetadataCatalog
if __name__ == '__main__':

    #Treinamento
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

    #Teste
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



#for d in random.sample(dataset_dicts, 1):
#    img = cv2.imread(d["file_name"])
#    v = Visualizer(img[:, :, ::-1], metadata=frutas_metadata, scale=0.5)
#    v = v.draw_dataset_dict(d)
#    matplotlib.use('tkagg')
#    plt.figure(figsize = (5, 5))
#    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#    plt.show()
