import typer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, create_dataloader
from pytorch_lightning import seed_everything
from typing import Optional

app = typer.Typer()

def train_model(
    config: str,
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

    # First use CPU to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config).cpu()

    if resume_path is not None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))

    model.sd_locked = sd_locked
    model.learning_rate = learning_rate

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/DODA',
        filename='{step}',
        save_weights_only=True,
        save_top_k=1,  # Only save the latest checkpoint
    )

    logger = ImageLogger(batch_frequency=logger_freq)

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        callbacks=[logger, checkpoint_callback],
        accumulate_grad_batches=accumulate_grad_batches,
        max_steps=max_steps
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


@app.command()
def vae(resume: Optional[str] = typer.Option(None, help="Ścieżka do checkpointu")):
    config = 'configs/DODA/DODA_wheat_vae.yaml'
    train_model(config=config, resume_path=resume)


@app.command()
def ldm(resume: Optional[str] = typer.Option(None, help="Ścieżka do checkpointu")):
    config = 'configs/DODA/DODA_wheat_ldm_kl_4_layout_clip.yaml'
    train_model(config=config, resume_path=resume)


@app.command()
def l2i(resume: Optional[str] = typer.Option(None, help="Ścieżka do checkpointu")):
    config = 'configs/DODA/DODA_wheat_ldm_img2img.yaml'
    train_model(config=config, resume_path=resume)


if __name__ == "__main__":
    app()
