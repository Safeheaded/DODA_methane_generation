import os
from typing import Optional

import typer
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, create_dataloader
from pytorch_lightning import seed_everything

# Load .env variables
load_dotenv()

# Neptune credentials from .env
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_KEY")
PROJECTS = {
    "vae": os.getenv("NEPTUNE_VAE_PROJECTNAME"),
    "ldm": os.getenv("NEPTUNE_LDM_PROJECTNAME"),
    "l2i": os.getenv("NEPTUNE_L2I_PROJECTNAME"),
}

# Typer app
app = typer.Typer()


def train_model(
    config: str,
    project_key: str,
    resume_path: Optional[str] = None,
    logger_freq: int = 5000,
    max_steps: int = 80000,
    sd_locked: bool = True,
    learning_rate: float = 1e-5,
    accumulate_grad_batches: int = 1,
    seed: int = 23
):
    seed_everything(seed)

    train_dataloader, val_dataloader = create_dataloader(config)

    model = create_model(config).cpu()

    if resume_path is not None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))

    model.sd_locked = sd_locked
    model.learning_rate = learning_rate

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/DODA',
        filename='{step}',
        save_weights_only=True,
        save_top_k=1,
    )

    image_logger = ImageLogger(batch_frequency=logger_freq)

    # Neptune logger
    if NEPTUNE_API_TOKEN is None or PROJECTS[project_key] is None:
        raise ValueError(f"Missing NEPTUNE_API_TOKEN or project name for key '{project_key}' in .env")

    neptune_logger = NeptuneLogger(
        api_key=NEPTUNE_API_TOKEN,
        project=PROJECTS[project_key],
        name=f"DODA-{project_key}",
        log_model_checkpoints=False,
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        callbacks=[image_logger, checkpoint_callback],
        logger=[neptune_logger],
        accumulate_grad_batches=accumulate_grad_batches,
        max_steps=max_steps
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


@app.command()
def vae(resume: Optional[str] = typer.Option(None, help="Ścieżka do checkpointu")):
    config = 'configs/DODA/DODA_wheat_vae.yaml'
    train_model(config=config, project_key="vae", resume_path=resume)


@app.command()
def ldm(resume: Optional[str] = typer.Option(None, help="Ścieżka do checkpointu")):
    config = 'configs/DODA/DODA_wheat_ldm_kl_4_layout_clip.yaml'
    train_model(config=config, project_key="ldm", resume_path=resume)


@app.command()
def l2i(resume: Optional[str] = typer.Option(None, help="Ścieżka do checkpointu")):
    config = 'configs/DODA/DODA_wheat_ldm_img2img.yaml'
    train_model(config=config, project_key="l2i", resume_path=resume)


if __name__ == "__main__":
    app()
