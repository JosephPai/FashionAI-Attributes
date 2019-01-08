import os

# 4 gpus
gpu_list = [0, 1, 2, 3]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

task_name = "pant_length"   # choose from task_list.keys()
model_name = "weights/model_%s.h5" % task_name
image_size = 448

TRAIN_PATH = r"datasets/Annotations/train/%s.csv"
TEST_PATH = r"datasets/Annotations/test/%s.csv"
SAVE_LABEL_PATH = r"result/result_%s.txt" % task_name

task_list = {
    'pant_length': ["Invisible", "Short Pant", "Mid Length", "3/4 Length", "Cropped Pant", "Full Length"],
    'skirt_length': ["Invisible", "Short Length", "Knee Length", "Midi Length", "Ankle Length", "Floor Length"],
    'sleeve_length': ["Invisible", "Sleeveless", "Cup Sleeves", "Short Sleeves", "Elbow Sleeves",
                      "3/4 Sleeves", "Wrist Length", "Long Sleeves", "Extra Long Sleeves"],
    'coat_length': ["Invisible", "High Waist Length", "Regular Length", "Long Length",
                    "Micro Length", "Knee Length", "Midi Length", "Ankle&Floor Length"],
    'collar_design': ["Invisible", "Shirt Collar", "Peter Pan", "Puritan Collar", "Rib Collar"],
    'lapel_design': ["Invisible", "Notched", "Collarless", "Shawl Collar", "Plus Size Shawl"],
    'neckline_design': ["Invisible", "Strapless Neck", "Deep V Neckline", "Straight Neck", "V Neckline",
                        "Square Neckline", "Off Shoulder", "Round Neckline",
                        "Sweat Heart Neck", "One	Shoulder Neckline"],
    'neck_design': ["Invisible", "Turtle Neck", "Ruffle Semi-High Collar", "Low Turtle Neck", "Draped Collar"],
}


