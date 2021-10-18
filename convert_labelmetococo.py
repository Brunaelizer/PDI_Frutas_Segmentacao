import labelme2coco

labelme_folder = "data/Teste/Train"

# set path for coco json to be saved
save_json_path = "data/Teste/Train/test_coco.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)