import atexit

import numpy as np
import wandb

from settings_module import Settings
from setup.logging_utils import RunningMeanStack

# if program crashes or is killed, make sure to save the run
atexit.register(wandb.finish)


class WandbModule:
    """
    This class creates a wandb run and saves the config.
    """

    def __init__(self, settings: Settings):
        self.run = wandb.init(
            project=settings.project,
            group=settings.group,
            name=settings.name,
            notes=settings.notes,
            resume=settings.resume,
            mode=settings.mode,
            id=settings.id,
            config=settings.cfg_dict
        )

        self.run.log_code(".")
        self.model_log = wandb.Artifact('xai-cyclegan_model', type='model', description=settings.model_notes)

        self.step = 0
        self.log_frequency = settings.log_frequency

        self.generator_he_to_p63_running_loss_avg = RunningMeanStack(self.log_frequency)
        self.generator_p63_to_he_running_loss_avg = RunningMeanStack(self.log_frequency)
        self.discriminator_he_running_loss_avg = RunningMeanStack(self.log_frequency)
        self.discriminator_p63_running_loss_avg = RunningMeanStack(self.log_frequency)
        self.cycle_he_running_loss_avg = RunningMeanStack(self.log_frequency)
        self.cycle_p63_running_loss_avg = RunningMeanStack(self.log_frequency)
        self.total_running_loss_avg = RunningMeanStack(self.log_frequency)

    def log(self, epoch):
        self.run.log({
            "he_to_p63_generator_loss": self.generator_he_to_p63_running_loss_avg.mean,
            "p63_to_he_generator_loss": self.generator_p63_to_he_running_loss_avg.mean,
            "he_discriminator_loss": self.discriminator_he_running_loss_avg.mean,
            "p63_discriminator_loss": self.discriminator_p63_running_loss_avg.mean,
            "he_cycle_loss": self.cycle_he_running_loss_avg.mean,
            "p63_cycle_loss": self.cycle_p63_running_loss_avg.mean,
            "epoch": epoch,
        }, step=self.step)

    def log_image(self, real_image_pair, gen_image_pair, recon_image_pair):

        real_image_pair = [real_image_pair[0].cpu().detach().numpy(), real_image_pair[1].cpu().detach().numpy()]
        gen_image_pair = [gen_image_pair[0].cpu().detach().numpy(), gen_image_pair[1].cpu().detach().numpy()]
        recon_image_pair = [recon_image_pair[0].cpu().detach().numpy(), recon_image_pair[1].cpu().detach().numpy()]

        real_image = np.concatenate((real_image_pair[0], real_image_pair[1]), axis=2)
        gen_image = np.concatenate((gen_image_pair[0], gen_image_pair[1]), axis=2)
        recon_image = np.concatenate((recon_image_pair[0], recon_image_pair[1]), axis=2)
        real_image = np.concatenate((real_image, gen_image), axis=1)
        real_image = np.concatenate((real_image, recon_image), axis=1)

        self.run.log({
            "generation_results": real_image,
        }, step=self.step)

    def log_model(self, model_file):
        self.model_log.add_file(model_file)
        self.run.log_artifact(self.model_log)
