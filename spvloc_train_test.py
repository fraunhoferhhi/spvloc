import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if not os.name == "nt":
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pytorch_lightning as pl
import torch
import glob

from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from spvloc.config.defaults import get_cfg_defaults
from spvloc.config.parser import parse_args
from spvloc.model.spvloc import PerspectiveImageFromLayout


def load_test_checkpoint(checkpoint_path, model):
    if not checkpoint_path.endswith(".ckpt"):  # If it's a directory
        # Use the code to find and load the newest checkpoint within the directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.ckpt"))
        sorted_checkpoints = sorted(checkpoint_files)
        checkpoint_path = sorted_checkpoints[-1]
    print("Load test checkpoint ", checkpoint_path)
    load = torch.load(checkpoint_path)
    # Change strict to false if something has changed about the model.
    model.load_state_dict(load["state_dict"], strict=True)


if __name__ == "__main__":
    args = parse_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.freeze()

    pl.seed_everything(config.SEED)

    if args.checkpoint_file:
        resume_path = args.checkpoint_file
    else:
        resume_path = None

    model = PerspectiveImageFromLayout(config)

    logger = loggers.TensorBoardLogger(config.OUT_DIR, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(save_top_k=1)

    log_inverval = 1 if args.test_ckpt else 20

    trainer = pl.Trainer(
        max_epochs=config.TRAIN.NUM_EPOCHS,
        devices=config.SYSTEM.NUM_GPUS,
        accelerator="auto",
        logger=logger,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.TRAIN.TEST_EVERY,
        log_every_n_steps=log_inverval,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=log_inverval)],
    )

    if args.test_ckpt:
        assert config.SYSTEM.NUM_GPUS == 1
        load_test_checkpoint(args.test_ckpt, model)
        trainer.test(model)
    else:
        # add this to load an old model, if some new parameters have been added, optimizer info is lost
        # load = torch.load(resume_path)
        # model.load_state_dict(load["state_dict"], strict=False)
        # trainer.fit(model)

        trainer.fit(model, ckpt_path=resume_path)
