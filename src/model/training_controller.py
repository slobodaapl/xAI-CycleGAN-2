import itertools
from typing import Callable

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim

from model.dataset import DatasetFromFolder
from model.explanation import ExplanationController
from model.mask import get_mask
from model.model import Generator, Discriminator
from model.utils import LambdaLR, ImagePool

from setup.settings_module import Settings
from setup.wandb_module import WandbModule


L_RANGE = 1.68976005407


def clear_nan_hook(module, grad_i, grad_o):
    grad_i = torch.nan_to_num(grad_i[0], nan=0.0, posinf=0.0, neginf=0.0)
    return module, grad_i, grad_o


class TrainingController:

    def __init__(self, settings: Settings, wandb_module: WandbModule):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.settings = settings
        self.wandb_module = wandb_module
        
        self.latest_generator_loss = None
        self.latest_discriminator_he_loss = None
        self.latest_discriminator_p63_loss= None
        self.latest_identity_loss = None
        self.latest_cycle_loss = None
        self.latest_ssim_loss = None

        # region Initialize data loaders
        self.train_he_data = DatasetFromFolder(settings.data_root, settings.data_train_he, settings.norm_dict)
        self.train_he = DataLoader(dataset=self.train_he_data, batch_size=settings.batch_size, shuffle=True)

        self.train_p63_data = DatasetFromFolder(settings.data_root, settings.data_train_p63, settings.norm_dict)
        self.train_p63 = DataLoader(dataset=self.train_p63_data, batch_size=settings.batch_size, shuffle=True)

        self.test_he_data = DatasetFromFolder(settings.data_root, settings.data_test_he, settings.norm_dict)
        self.test_he = DataLoader(dataset=self.test_he_data, batch_size=settings.batch_size, shuffle=False)

        self.test_p63_data = DatasetFromFolder(settings.data_root, settings.data_test_p63, settings.norm_dict)
        self.test_p63 = DataLoader(dataset=self.test_p63_data, batch_size=settings.batch_size, shuffle=False)
        # endregion

        # region Initialize models
        generator_params = (settings.generator_downconv_filters, settings.num_resnet_blocks, settings.channels, settings.channels)
        discriminator_params = (settings.discriminator_downconv_filters, settings.channels)
        self.generator_he_to_p63 = Generator(*generator_params)
        self.generator_p63_to_he = Generator(*generator_params)
        self.discriminator_he = Discriminator(*discriminator_params)
        self.discriminator_p63 = Discriminator(*discriminator_params)
        self.discriminator_he_mask = Discriminator(*discriminator_params)
        self.discriminator_p63_mask = Discriminator(*discriminator_params)

        self.generator_he_to_p63.to(self.device)
        self.generator_p63_to_he.to(self.device)
        self.discriminator_he.to(self.device)
        self.discriminator_p63.to(self.device)
        self.discriminator_he_mask.to(self.device)
        self.discriminator_p63_mask.to(self.device)
        # endregion

        # region Initialize nan_removal_hooks
        self.generator_he_to_p63.final.register_backward_hook(clear_nan_hook)
        self.generator_p63_to_he.final.register_backward_hook(clear_nan_hook)
        self.discriminator_he.conv7.register_backward_hook(clear_nan_hook)
        self.discriminator_p63.conv7.register_backward_hook(clear_nan_hook)
        self.discriminator_he_mask.conv7.register_backward_hook(clear_nan_hook)
        self.discriminator_p63_mask.conv7.register_backward_hook(clear_nan_hook)
        # endregion

        # region Initialize wandb model watching
        self.wandb_module.run.watch(
            (
                self.generator_he_to_p63,
                self.generator_p63_to_he,
                self.discriminator_he,
                self.discriminator_he_mask,
                self.discriminator_p63,
                self.discriminator_p63_mask
            ),
            log="all",
            log_freq=wandb_module.log_frequency,
            log_graph=True
        )
        # endregion

        # region Initialize explanation classes
        self.he_explainer = ExplanationController(
            self.discriminator_he.loss_fake,
            self.discriminator_he_mask.loss_fake,
            settings.lambda_mask_adversarial_ratio,
            settings.explanation_ramp_type
        )

        self.p63_explainer = ExplanationController(
            self.discriminator_p63.loss_fake,
            self.discriminator_p63_mask.loss_fake,
            settings.lambda_mask_adversarial_ratio,
            settings.explanation_ramp_type
        )

        # P63 explainer contains P63 discriminator, used to explain P63 generation mistakes to HE gen.
        # Therefore, we have to assign P63 explainer to the HE to P63 generator.
        self.generator_he_to_p63.final.register_backward_hook(self.p63_explainer.explanation_hook)
        self.generator_p63_to_he.final.register_backward_hook(self.he_explainer.explanation_hook)
        # endregion

        # region Initialize loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixel_wise = torch.nn.L1Loss()
        # endregion

        # region Initialize optimizers
        discriminator_he_params = itertools.chain(
            self.discriminator_he.parameters(),
            self.discriminator_he_mask.parameters()
        )

        discriminator_p63_params = itertools.chain(
            self.discriminator_p63.parameters(),
            self.discriminator_p63_mask.parameters()
        )

        self.generator_optimizer = torch.optim.Adam(
            itertools.chain(self.generator_he_to_p63.parameters(), self.generator_p63_to_he.parameters()),
            lr=settings.lr_generator, betas=(settings.beta1, settings.beta2)
        )

        self.discriminator_he_optimizer = torch.optim.Adam(
            discriminator_he_params,
            lr=settings.lr_discriminator, betas=(settings.beta1, settings.beta2)
        )

        self.discriminator_p63_optimizer = torch.optim.Adam(
            discriminator_p63_params,
            lr=settings.lr_discriminator, betas=(settings.beta1, settings.beta2)
        )

        self.lr_generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer, lr_lambda=LambdaLR(settings.epochs, settings.decay_epoch).step
        )

        self.lr_discriminator_he_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.discriminator_he_optimizer, lr_lambda=LambdaLR(settings.epochs, settings.decay_epoch).step
        )

        self.lr_discriminator_p63_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.discriminator_p63_optimizer, lr_lambda=LambdaLR(settings.epochs, settings.decay_epoch).step
        )
        # endregion

        # region Initialize image pool
        pool_size = settings.pool_size
        self.fake_he_pool = ImagePool(pool_size)
        self.fake_p63_pool = ImagePool(pool_size)
        # endregion

    def get_loss(self, tensor: torch.Tensor, loss_function: Callable, target_function: Callable) -> torch.Tensor:
        return loss_function(tensor, Variable(target_function(tensor.size()).to(self.device)))
    
    def get_total_mask_disc_loss(self, real: torch.Tensor, mask: torch.Tensor,
                                 fake: torch.Tensor, discriminator_mask: Discriminator) -> torch.Tensor:

        discriminator_mask_real_decision = discriminator_mask(real * mask)
        discriminator_mask_real_loss = \
            self.get_loss(discriminator_mask_real_decision, self.criterion_GAN, torch.ones)
        discriminator_mask_fake_decision = \
            discriminator_mask(fake * mask)
        discriminator_mask_fake_loss = \
            self.get_loss(discriminator_mask_fake_decision, self.criterion_GAN, torch.zeros)

        return discriminator_mask_real_loss + discriminator_mask_fake_loss

    def get_total_gen_loss_and_prep_explainer(self, real: torch.Tensor, mask: torch.Tensor,
                                              generator: Generator,
                                              discriminator: Discriminator,
                                              explainer: ExplanationController) -> torch.Tensor:

        fake = generator(real, mask)
        disc_fake = discriminator(fake)
        disc_fake_mask = self.discriminator_p63_mask(fake * mask)
        generator_loss = self.get_loss(disc_fake, self.criterion_GAN, torch.ones)
        generator_mask_loss = self.get_loss(disc_fake_mask, self.criterion_GAN, torch.ones)

        explainer.set_explanation(fake)
        explainer.set_explanation_m(fake * mask)

        return (self.settings.lambda_mask_adversarial_ratio * generator_mask_loss
                + (1 - self.settings.lambda_mask_adversarial_ratio)
                * generator_loss) * self.settings.lambda_adversarial

    def get_total_cycle_loss(self, cycled: torch.Tensor, other_mask: torch.Tensor,
                             other_mask_inverted: torch.Tensor, other_real: torch.Tensor) -> torch.Tensor:

        pixel_wise_cycle_loss = self.criterion_pixel_wise(cycled * other_mask, other_real * other_mask)
        pixel_wise_cycle_loss_inv = self.criterion_pixel_wise(
            cycled * other_mask_inverted, other_real * other_mask_inverted)
        pixel_wise_cycle_loss = pixel_wise_cycle_loss * self.settings.lambda_mask_cycle_ratio
        pixel_wise_cycle_loss_inv = pixel_wise_cycle_loss_inv * (1 - self.settings.lambda_mask_cycle_ratio)

        return pixel_wise_cycle_loss + pixel_wise_cycle_loss_inv

    def get_partial_disc_loss(self, real: torch.Tensor, fake: torch.Tensor,
                              discriminator: Discriminator,
                              coefficient: float,
                              pool: ImagePool = None) -> torch.Tensor:

        discriminator_real_decision = discriminator(real)
        discriminator_real_loss = self.get_loss(discriminator_real_decision, self.criterion_GAN, torch.ones)

        if pool is not None:
            fake = pool.query(fake)

        discriminator_fake_decision = discriminator(fake)
        discriminator_fake_loss = self.get_loss(discriminator_fake_decision, self.criterion_GAN, torch.zeros)

        return (discriminator_real_loss + discriminator_fake_loss) * 0.5 * coefficient

    def training_step(self, real_he: torch.Tensor, real_p63: torch.Tensor):
        min_dim = min(real_he.size(0), real_p63.size(0))
        real_he = real_he[:min_dim]
        real_p63 = real_p63[:min_dim]

        mask_he = get_mask(real_he, self.settings.mask_type)
        mask_p63 = get_mask(real_p63, self.settings.mask_type)

        real_he: torch.Tensor = Variable(real_he.to(self.device))
        real_p63: torch.Tensor = Variable(real_p63.to(self.device))
        mask_he: torch.Tensor = Variable(mask_he.to(self.device))
        mask_p63: torch.Tensor = Variable(mask_p63.to(self.device))
        
        with torch.autocast(device_type="cuda"):
            fake_p63 = self.generator_he_to_p63(real_he, mask_he)
            fake_he = self.generator_p63_to_he(real_p63, mask_p63)
            cycled_he = self.generator_p63_to_he(fake_p63, mask_he)
            cycled_p63 = self.generator_he_to_p63(fake_he, mask_p63)

            self.p63_explainer.set_explanation_m(fake_p63 * mask_he)
            self.he_explainer.set_explanation_m(fake_he * mask_p63)

            with torch.no_grad():
                discriminator_he_mask_loss = \
                    self.get_total_mask_disc_loss(real_he, mask_he, fake_he, self.discriminator_he_mask) * 0.5
                discriminator_p63_mask_loss = \
                    self.get_total_mask_disc_loss(real_p63, mask_p63, fake_p63, self.discriminator_p63_mask) * 0.5

                mask_he = mask_he + self.p63_explainer.explanation_mask * self.p63_explainer.get_coefficient_mask(
                    discriminator_p63_mask_loss)  # maybe mask_he + mask_he * rest
                mask_p63 = mask_p63 + self.he_explainer.explanation_mask * self.he_explainer.get_coefficient_mask(
                    discriminator_he_mask_loss)

            he_mask_inverted = 1 - mask_he
            p63_mask_inverted = 1 - mask_p63
            he_mask_inverted: torch.Tensor = Variable(he_mask_inverted.to(self.device))
            p63_mask_inverted: torch.Tensor = Variable(p63_mask_inverted.to(self.device))

            # Train generator G
            # A -> B
            generator_he_to_p63_total_loss = self.get_total_gen_loss_and_prep_explainer(real_he,
                                                                                        mask_he,
                                                                                        self.generator_he_to_p63,
                                                                                        self.discriminator_p63,
                                                                                        self.p63_explainer)

            # forward cycle loss
            cycle_he_loss_total = self.get_total_cycle_loss(cycled_he, mask_he, he_mask_inverted, real_he)

            # B -> A
            generator_p63_to_he_total_loss = self.get_total_gen_loss_and_prep_explainer(real_p63,
                                                                                        mask_p63,
                                                                                        self.generator_p63_to_he,
                                                                                        self.discriminator_he,
                                                                                        self.he_explainer)

            # backward cycle loss
            cycle_p63_loss_total = self.get_total_cycle_loss(cycled_p63, mask_p63, p63_mask_inverted, real_p63)

            # total cycle loss
            cycle_loss = (cycle_he_loss_total + cycle_p63_loss_total) * self.settings.lambda_cycle

            # identity loss
            identity_he = self.criterion_pixel_wise(real_he, self.generator_p63_to_he(real_he, mask_he))
            identity_p63 = self.criterion_pixel_wise(real_p63, self.generator_he_to_p63(real_p63, mask_p63))
            identity_loss = (identity_he + identity_p63) * self.settings.lambda_identity

            # ssim loss
            ssim_he = ssim(real_he[:, 0:1, :, :] + L_RANGE, cycled_he[:, 0:1, :, :] + L_RANGE, data_range=L_RANGE*2)
            ssim_he_fake = ssim(real_he[:, 0:1, :, :] + L_RANGE, fake_p63[:, 0:1, :, :] + L_RANGE, data_range=L_RANGE*2)
            ssim_p63 = ssim(real_p63[:, 0:1, :, :] + L_RANGE, cycled_p63[:, 0:1, :, :] + L_RANGE, data_range=L_RANGE*2)
            ssim_p63_fake = ssim(real_p63[:, 0:1, :, :] + L_RANGE, fake_he[:, 0:1, :, :] + L_RANGE, data_range=L_RANGE*2)
            ssim_loss = (
                + (1 - ssim_he)
                + (1 - ssim_p63)
                + (1 - ssim_he_fake) * 0.75
                + (1 - ssim_p63_fake) * 0.75
            ) * self.settings.lambda_ssim * 0.25

            with torch.no_grad():
                discriminator_he_loss_partial = self.get_partial_disc_loss(real_he, fake_he,
                                                                           self.discriminator_he,
                                                                           1 - self.settings.
                                                                           lambda_mask_adversarial_ratio)

                discriminator_he_loss_mask_partial = self.get_partial_disc_loss(real_he * mask_he, fake_he * mask_p63,
                                                                                self.discriminator_he_mask,
                                                                                self.settings.
                                                                                lambda_mask_adversarial_ratio)

                discriminator_p63_loss_partial = self.get_partial_disc_loss(real_p63, fake_p63,
                                                                            self.discriminator_p63,
                                                                            1 - self.settings.
                                                                            lambda_mask_adversarial_ratio)

                discriminator_p63_loss_mask_partial = self.get_partial_disc_loss(real_p63 * mask_p63,
                                                                                 fake_p63 * mask_he,
                                                                                 self.discriminator_p63_mask,
                                                                                 self.settings.
                                                                                 lambda_mask_adversarial_ratio)

                self.p63_explainer.set_losses(discriminator_he_loss_partial, discriminator_he_loss_mask_partial)
                self.he_explainer.set_losses(discriminator_p63_loss_partial, discriminator_p63_loss_mask_partial)
                self.p63_explainer.get_explanation()
                self.he_explainer.get_explanation()

            # backward gen
            generator_loss = \
                + generator_he_to_p63_total_loss \
                + generator_p63_to_he_total_loss \
                + cycle_loss \
                + identity_loss \
                + ssim_loss

        self.generator_optimizer.zero_grad()
        generator_loss = torch.nan_to_num(generator_loss, nan=0, posinf=1, neginf=-1)
        generator_loss.backward()
        self.generator_optimizer.step()

        # Back propagation
        with torch.autocast(device_type="cuda"):
            discriminator_he_loss_partial = self.get_partial_disc_loss(real_he, fake_he, self.discriminator_he,
                                                                       1 - self.settings.lambda_mask_adversarial_ratio,
                                                                       self.fake_he_pool)

            discriminator_he_loss_mask_partial = self.get_partial_disc_loss(real_he * mask_he, fake_he * mask_p63,
                                                                            self.discriminator_he_mask,
                                                                            self.settings.lambda_mask_adversarial_ratio,
                                                                            self.fake_he_pool)

            discriminator_he_loss = discriminator_he_loss_partial + discriminator_he_loss_mask_partial

        self.discriminator_he_optimizer.zero_grad()
        discriminator_he_loss = torch.nan_to_num(discriminator_he_loss, nan=0, posinf=1, neginf=0)
        discriminator_he_loss.backward()
        self.discriminator_he_optimizer.step()

        with torch.autocast(device_type="cuda"):
            discriminator_p63_loss_partial = self.get_partial_disc_loss(real_p63, fake_p63, self.discriminator_p63,
                                                                        1 - self.settings.lambda_mask_adversarial_ratio,
                                                                        self.fake_p63_pool)

            discriminator_p63_loss_mask_partial = self.get_partial_disc_loss(real_p63 * mask_p63, fake_p63 * mask_he,
                                                                             self.discriminator_p63_mask,
                                                                             self.settings.lambda_mask_adversarial_ratio,
                                                                             self.fake_p63_pool)

            discriminator_p63_loss = discriminator_p63_loss_partial + discriminator_p63_loss_mask_partial

        self.discriminator_p63_optimizer.zero_grad()
        discriminator_p63_loss = torch.nan_to_num(discriminator_p63_loss, nan=0, posinf=1, neginf=0)
        discriminator_p63_loss.backward()
        self.discriminator_p63_optimizer.step()

        self.latest_generator_loss = generator_loss.item()
        self.latest_discriminator_he_loss = discriminator_he_loss.item()
        self.latest_discriminator_p63_loss = discriminator_p63_loss.item()
        self.latest_identity_loss = identity_loss.item()
        self.latest_cycle_loss = cycle_loss.item()
        self.latest_ssim_loss = ssim_loss.item()

        self.wandb_module.discriminator_he_running_loss_avg.append(discriminator_he_loss.item())
        self.wandb_module.discriminator_p63_running_loss_avg.append(discriminator_p63_loss.item())
        self.wandb_module.generator_he_to_p63_running_loss_avg.append(generator_he_to_p63_total_loss.item())
        self.wandb_module.generator_p63_to_he_running_loss_avg.append(generator_p63_to_he_total_loss.item())
        self.wandb_module.cycle_he_running_loss_avg.append(cycle_loss.item())
        self.wandb_module.cycle_p63_running_loss_avg.append(cycle_loss.item())
        self.wandb_module.total_running_loss_avg.append(generator_loss.item())

    def get_image_pairs(self):
        real_he = self.test_he_data.get_random_image()
        real_p63 = self.test_p63_data.get_random_image()

        real_he = Variable(real_he.to(self.device)).expand(1, -1, -1, -1)
        real_p63 = Variable(real_p63.to(self.device)).expand(1, -1, -1, -1)

        real_he_mask = get_mask(real_he, self.settings.mask_type)
        real_p63_mask = get_mask(real_p63, self.settings.mask_type)

        real_he_mask = Variable(real_he_mask.to(self.device))
        real_p63_mask = Variable(real_p63_mask.to(self.device))

        fake_p63 = self.generator_he_to_p63(real_he, real_he_mask)
        reconstructed_he = self.generator_p63_to_he(fake_p63, real_he_mask)

        fake_he = self.generator_p63_to_he(real_p63, real_p63_mask)
        reconstructed_p63 = self.generator_he_to_p63(fake_he, real_p63_mask)

        return (real_he, real_p63), (fake_he, fake_p63), (reconstructed_he, reconstructed_p63)