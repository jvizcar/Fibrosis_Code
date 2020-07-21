"""
Hard-coded variables that are used by the various scripts in the repo.
"""
# 0: Pathologists_Folders collection id, 1: Transplant Raw Slides collection id
DSA_PARENT_IDS = ['5e133f6cc1f21ef7e0e28cad', '5cf820a91ee779b5180282c4']

GLOM_GROUPS = ['Glomerulus_normal', 'Glomerulus_sclerotic', 'Glomerulus_inflammed']

# background class dirs will be automatically created
# make sure you manually create glomerulus dirs in the Train/Val/Test dirs by moving all the images in the GLOM_GROUPS
# dirs inside them. Currently the code does not have the capability of grouping tiles from different annotation groups
# together.
CLASS_NAMES = ['background', 'glomerulus']

TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15

MAG = 10  # magnification to pull images at
TILE_SIZE = (224, 224)

MODEL_VERSION = 'latest'  # latest uses the highest number model, otherwise specify a model version
SAVE_NAME = 'PAS_GLOM'  # prepend name for annotations
