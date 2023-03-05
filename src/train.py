import os

import torch

from model.training_controller import TrainingController
from setup.settings_module import Settings
from setup.wandb_module import WandbModule

settings = Settings('settings.cfg')
wandb_module = WandbModule(settings)
training_controller = TrainingController(settings, wandb_module)

# Directories for loading data and saving results
data_dir = settings.data_root

os.mkdir('models') if not os.path.exists('models') else None
model_dir = os.path.join(os.getcwd(), 'models')

os.mkdir(f'{settings.id}') if not os.path.exists(f'{settings.id}') else None
model_dir = os.path.join(model_dir, f'{settings.id}')
model_file = os.path.join(model_dir, f'model_checkpoint.pth')


for epoch in range(settings.epochs):
    for step, (real_he, real_p63) in enumerate(zip(training_controller.train_he, training_controller.train_p63)):
        training_controller.training_step(real_he, real_p63)

        if step % settings.log_frequency == 0:
            wandb_module.log(epoch)
            wandb_module.log_image(*training_controller.get_image_pairs())
            wandb_module.step += 1

    training_controller.lr_generator_scheduler.step()
    training_controller.lr_discriminator_he_scheduler.step()
    training_controller.lr_discriminator_p63_scheduler.step()

    if epoch % settings.checkpoint_frequency_epochs == 0:
        torch.save({
            'epoch': epoch,
            'generator_he_to_p63_state_dict': training_controller.generator_he_to_p63.state_dict(),
            'generator_p63_to_he_state_dict': training_controller.generator_p63_to_he.state_dict(),
            'discriminator_he_state_dict': training_controller.discriminator_he.state_dict(),
            'discriminator_p63_state_dict': training_controller.discriminator_p63.state_dict(),
            'generator_optimizer_state_dict': training_controller.generator_optimizer.state_dict(),
            'discriminator_he_optimizer_state_dict': training_controller.discriminator_he_optimizer.state_dict(),
            'discriminator_p63_optimizer_state_dict': training_controller.discriminator_p63_optimizer.state_dict(),
            'generator_loss': training_controller.latest_generator_loss,
            'discriminator_he_loss': training_controller.latest_discriminator_he_loss,
            'discriminator_p63_loss': training_controller.latest_discriminator_p63_loss,
        }, f=model_file)

        wandb_module.log_model(model_file)
