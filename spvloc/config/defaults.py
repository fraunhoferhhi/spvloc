from yacs.config import CfgNode as CN

_C = CN()

_C.OUT_DIR = "./runs"  # Output directory for saving the results or logs
_C.SEED = 42  # Seed for reproducibility
_C.CONSOLE_LOG_INVERVAL = 10  # Interval for logging messages to the console

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 0  # Number of workers for data loading
_C.SYSTEM.NUM_GPUS = 1  # Number of GPUs to use during training (untested for more than 1)

_C.INPUT = CN()
_C.INPUT.IMG_SIZE = (256, 256)  # Input image size (height, width)
_C.INPUT.NORMALISE_MEAN = [0.485, 0.456, 0.406]  # Normalization mean values for image preprocessing
_C.INPUT.NORMALISE_STD = [0.229, 0.224, 0.225]  # Normalization standard deviation values

_C.DATASET = CN()
_C.DATASET.PATH = "G:/Datasets/Structured3D/Structured3D_bbox/Structured3D"  # Dataset path
_C.DATASET.PREPARED_DATA_PATH = "pickle"  # Folder to save scenes as pickle
_C.DATASET.PREPARED_DATA_PATH_TEST = "pickle_test"  # Folder to save test batches as pickle
_C.DATASET.USE_PREPARED_DATA = False  # Use pickled scenes (quickens data loading)
_C.DATASET.CACHE_TEST_BATCHES = False  # Save batches as pickle
_C.DATASET.TEST_LIGHTING = ["raw"]  # ight mode during training (always raw for zillow)
_C.DATASET.TEST_SET_FURNITURE = ["full"]  # Furniture config during testing
_C.DATASET.TRAIN_LIGHTING = ["raw"]  # Light mode during training (always raw for zillow)
_C.DATASET.TRAIN_FURNITURE = ["empty", "full"]  # Furniture config during trainig
_C.DATASET.AUGMENT_QUERYS = True  # Apply some simple color and blur augmentation
_C.DATASET.PERSP_FROM_PANO = False  # Sample perspective train images from panoramas
_C.DATASET.PERSP_FROM_PANO_FOV = 80.0  # Field of view during training, ignored if LEARN_FOV = True
_C.DATASET.PERSP_FROM_PANO_RANGE = [180, 10, 10]  # Sample ranges (plusminus) for pranorama sampling
_C.DATASET.FILTER_EMPTY_IMAGES = True  # Fiter perspective images containing only one semantic class
# Rather look inside the room than just in a random direction during training
_C.DATASET.PERSP_FROM_PANO_FILTER_VALID_YAW = True
# Filter samples where the minimal distance to cam is smaller than this value (in m)
_C.DATASET.MIN_AVERAGE_DISTANCE = 0.0
_C.DATASET.NAME = "S3D"  # Dataset name (Zillow or S3D)
# This should only be True, if the perspective images of S3D have not been downloaded
_C.DATASET.S3D_NO_PERSP_IMAGES = False

_C.RENDER = CN()
_C.RENDER.IMG_SIZE = (256, 256)  # Rendered image size
_C.RENDER.PANO_SIZE = (128, 256)  # Rendered pano size
_C.RENDER.FIX_SEED = False  # Fix render seed (should be used during testing for deterministic rendering)

_C.MODEL = CN()
_C.MODEL.IMAGE_BACKBONE = "resnet18"  # Backbone for camera image encoding (resnet18, efficient_s or efficient_s_skip)
_C.MODEL.DESC_LENGTH = 256  # Embedding dim of encoder decoder bottleneck
_C.MODEL.NORMALISE_EMBEDDING = True  # Normalise bottleneck features
_C.MODEL.EMBEDDER_TYPE_IMAGE_MODULE = "fc"  # Bottleneck type (fc, mlp, spatial)
_C.MODEL.DECODER_RESOLUTION = (256, 256)  # Output size of layout decoding
_C.MODEL.DECODER_SEMANTIC_CLASSES = 6  # No of semantic classes (six in paper usecase)
_C.MODEL.PANO_BB_EXTENSION_FACTOR = 1.2  # Size extension of features to better attatch pose head
_C.MODEL.PANO_USE_EQUICONV = False  # Use equiconv during pano processing (ablation study, removed code)
_C.MODEL.PREDICT_POSES = True  # Train to predict realtive pose offsets
_C.MODEL.PANO_ENCODE_DEPTH = False  # Encode rendered panorama depth
_C.MODEL.PANO_ENCODE_NORMALS = True  # Encode rendered panorama normals
_C.MODEL.PANO_ENCODE_SEMANTICS = True  # Encode rendered panorama semantics
_C.MODEL.NORMALIZE_PANO_INPUT = False  # Normalise input panorama features
_C.MODEL.PANO_MAX_DEPTH = 10.0  # Maximum depth in input panorama (the rest is clipped)
_C.MODEL.LEARN_FOV = False  # Train model with random fov between 45 and 135. Fov as additional norwork input.
_C.MODEL.LEARN_ASPECT = False  # Train model with random aspect radios (padded)

_C.POSE_REFINE = CN()
_C.POSE_REFINE.MAX_ITERS = 1  # Refinement iterations during testing (0 means no refinement)

_C.TRAIN = CN()
_C.TRAIN.NUM_EPOCHS = 10  # Number of epochs
_C.TRAIN.BATCH_SIZE = 16  # Batch size (query images per batch)
_C.TRAIN.TEST_EVERY = 1  # Validation interval
_C.TRAIN.INITIAL_LR = 0.01  # Initial learning rate
_C.TRAIN.LR_MILESTONES = [5, 8]  # Multiply lr with LR_GAMMA after these epochs
_C.TRAIN.LR_GAMMA = 0.5  # Learning rate reduction factor
_C.TRAIN.NUM_NEAR_SAMPLES = 1  # Number of near panorama samples during training
_C.TRAIN.NUM_FAR_SAMPLES = 0  # Number of far panorama samples during training
_C.TRAIN.PANO_POSE_OFFSETS = [1.4, 1.4, 0.3]  # Ranges in which near panorama images are sampled.
_C.TRAIN.RETRAIN_DECODER = False  # If this is False, the query decoder is ignored
_C.TRAIN.FILTER_PANO_SAMPLES = False  # Apply some tests to filter out potentially ambiguous train images
_C.TRAIN.PERSP_FROM_PANO_RANGE_OPTIMIZE_EULER = False  # Predict euler angles instead quaternion (untested)
_C.TRAIN.IGNORE_MASK_BCE = False  # Ignore buinary cross entropy loss, viewport mask prediction

_C.TEST = CN()
_C.TEST.IGNORE_PRETRAINED = False  # No pretrained weights in encoders (if True)
_C.TEST.POSE_SAMPLE_STEP = 500  # Step size for sampling positions during testing
_C.TEST.ADVANCED_POSE_SAMPLING = False  # Whether to use advanced position sampling techniques
_C.TEST.VAL_AS_TEST = False  # Whether to use validation data as test data
_C.TEST.CAMERA_HEIGHT = 1400  # Height of the camera during testing (in millimeters)
_C.TEST.PLOT_OUTPUT = False  # Whether to plot the output results during testing
_C.TEST.SAVE_PLOT_OUTPUT = False  # Whether to save the plotted output results
_C.TEST.SAVE_PLOT_DETAILS = False  # Whether to save detailed plots
_C.TEST.EVAL_GT_MAP_POINT = False  # Whether to check pose erorr against closest reference panorama
_C.TEST.EVAL_TOP3 = True  # Whether to check pose erorr for top 3 or only for top 1 match

_C.CONVERSION = CN()
_C.CONVERSION.INPUT_PATH = ""
_C.CONVERSION.OUTPUT_PATH = ""
_C.CONVERSION.SAMPLE_FOV = [60, 80, 90, 120]
_C.CONVERSION.ANGLE_OFFSET = [(0, 0), (5, 5), (10, 10), (15, 15), (20, 20)]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
