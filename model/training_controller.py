import itertools
from typing import TYPE_CHECKING, Callable

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.dataset import DatasetFromFolder
from model.explanation import ExplanationController
from model.mask import get_mask
from model.model import Generator, Discriminator
from model.utils import LambdaLR, ImagePool

if TYPE_CHECKING:
    from setup.settings_module import Settings
    from setup.wandb_module import WandbModule


class TrainingController:

    def __init__(self, settings: Settings, wandb_module: WandbModule):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.settings = settings
        self.wandb_module = wandb_module

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
        generator_params = (settings.channels, settings.generator_downconv_filters, settings.num_resnet_blocks)
        discriminator_params = (settings.channels, settings.discriminator_downconv_filters)
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
        self.generator_he_to_p63.deconv4.register_backward_hook(self.p63_explainer.explanation_hook)
        self.generator_p63_to_he.deconv4.register_backward_hook(self.he_explainer.explanation_hook)
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

        self.latest_generator_loss = None
        self.latest_discriminator_he_loss = None
        self.latest_discriminator_p63_loss = None

    def get_loss(self, tensor: torch.Tensor, loss_function: Callable, target_function: Callable) -> torch.Tensor:
        return loss_function(tensor, Variable(target_function(tensor.size()).to(self.device)))

    def training_step(self, real_he: torch.Tensor, real_p63: torch.Tensor):
        min_dim = min(real_he.size(0), real_p63.size(0))
        real_he = real_he[:min_dim]
        real_p63 = real_p63[:min_dim]

        mask_he = get_mask(real_he, self.settings.mask_type)
        mask_p63 = get_mask(real_p63, self.settings.mask_type)

        real_he = Variable(real_he.to(self.device))
        real_p63 = Variable(real_p63.to(self.device))
        mask_he = Variable(mask_he.to(self.device))
        mask_p63 = Variable(mask_p63.to(self.device))

        with torch.no_grad():
            fake_p63 = self.generator_he_to_p63(real_he, mask_he)
            fake_he = self.generator_p63_to_he(real_p63, mask_p63)

        self.p63_explainer.set_explanation_m(fake_p63 * mask_p63)
        self.he_explainer.set_explanation_m(fake_he * mask_he)

        with torch.no_grad():
            discriminator_he_mask_real_decision = self.discriminator_he_mask(real_he * mask_he)
            discriminator_he_mask_real_loss = \
                self.get_loss(discriminator_he_mask_real_decision, self.criterion_GAN, torch.ones)
            discriminator_he_mask_fake_decision = \
                self.discriminator_he_mask(fake_he * mask_he)
            discriminator_he_mask_fake_loss = \
                self.get_loss(discriminator_he_mask_fake_decision, self.criterion_GAN, torch.zeros)

            discriminator_p63_mask_real_decision = self.discriminator_p63_mask(real_p63 * mask_p63)
            discriminator_p63_mask_real_loss = self.get_loss(discriminator_p63_mask_real_decision, self.criterion_GAN,
                                                             torch.ones)
            discriminator_p63_mask_fake_decision = self.discriminator_p63_mask(fake_p63 * mask_p63)
            discriminator_p63_mask_fake_loss = self.get_loss(discriminator_p63_mask_fake_decision, self.criterion_GAN,
                                                             torch.zeros)

            discriminator_he_mask_loss = (discriminator_he_mask_real_loss + discriminator_he_mask_fake_loss) * 0.5
            discriminator_p63_mask_loss = (discriminator_p63_mask_real_loss + discriminator_p63_mask_fake_loss) * 0.5

            mask_he = mask_he + self.p63_explainer.explanation_mask * self.p63_explainer.get_coefficient_mask(
                discriminator_p63_mask_loss)  # maybe mask_he + mask_he * rest
            mask_p63 = mask_p63 + self.he_explainer.explanation_mask * self.he_explainer.get_coefficient_mask(
                discriminator_he_mask_loss)

        he_mask_inverted = 1 - mask_he
        p63_mask_inverted = 1 - mask_p63
        he_mask_inverted = Variable(he_mask_inverted.to(self.device))
        p63_mask_inverted = Variable(p63_mask_inverted.to(self.device))

        # Train generator G
        # A -> B
        fake_p63 = self.generator_he_to_p63(real_he, mask_he)
        discriminator_p63_fake_decision = self.discriminator_p63(fake_p63)
        discriminator_p63_mask_fake_decision = self.discriminator_p63_mask(fake_p63 * mask_p63)
        generator_he_to_p63_loss = self.get_loss(discriminator_p63_fake_decision, self.criterion_GAN, torch.ones)
        generator_he_to_p63_mask_loss = self.get_loss(
            discriminator_p63_mask_fake_decision, self.criterion_GAN, torch.ones
        )
        generator_he_to_p63_total_loss = (
            self.settings.lambda_mask_adversarial_ratio * generator_he_to_p63_mask_loss
            + (1 - self.settings.lambda_mask_adversarial_ratio)
            * generator_he_to_p63_loss) * self.settings.lambda_adversarial

        self.p63_explainer.set_explanation(fake_p63)
        self.p63_explainer.set_explanation_m(fake_p63 * mask_p63)

        # forward cycle loss
        he_reconstructed = self.generator_p63_to_he(fake_p63, mask_he)
        pixelwise_cycle_loss_he = self.criterion_pixel_wise(he_reconstructed * mask_he, real_he * mask_he)
        pixelwise_cycle_loss_he_inv = self.criterion_pixel_wise(
            he_reconstructed * he_mask_inverted, real_he * he_mask_inverted)
        pixelwise_cycle_loss_he = pixelwise_cycle_loss_he * self.settings.lambda_mask_cycle_ratio
        pixelwise_cycle_loss_he_inv = pixelwise_cycle_loss_he_inv * (1 - self.settings.lambda_mask_cycle_ratio)
        cycle_he_loss_total = pixelwise_cycle_loss_he + pixelwise_cycle_loss_he_inv

        # B -> A
        fake_he = self.generator_p63_to_he(real_p63, mask_p63)
        discriminator_he_fake_decision = self.discriminator_he(fake_he)
        discriminator_he_mask_fake_decision = self.discriminator_he_mask(fake_he * mask_he)
        generator_p63_to_he_loss = self.get_loss(discriminator_he_fake_decision, self.criterion_GAN, torch.ones)
        generator_p63_to_he_mask_loss = self.get_loss(
            discriminator_he_mask_fake_decision, self.criterion_GAN, torch.ones)
        generator_p63_to_he_total_loss = (
            self.settings.lambda_mask_adversarial_ratio * generator_p63_to_he_mask_loss
            + (1 - self.settings.lambda_mask_adversarial_ratio)
            * generator_p63_to_he_loss
        ) * self.settings.lambda_adversarial

        self.he_explainer.set_explanation(fake_he)
        self.he_explainer.set_explanation_m(fake_he * mask_he)

        # backward cycle loss
        p63_reconstructed = self.generator_he_to_p63(fake_he, mask_p63)
        pixelwise_cycle_loss_p63 = self.criterion_pixel_wise(p63_reconstructed * mask_p63, real_p63 * mask_p63)
        pixelwise_cycle_loss_p63_inv = self.criterion_pixel_wise(p63_reconstructed * p63_mask_inverted,
                                                                 real_p63 * p63_mask_inverted)
        pixelwise_cycle_loss_p63 = pixelwise_cycle_loss_p63 * self.settings.lambda_mask_cycle_ratio
        pixelwise_cycle_loss_p63_inv = pixelwise_cycle_loss_p63_inv * (1 - self.settings.lambda_mask_cycle_ratio)
        cycle_p63_loss_total = pixelwise_cycle_loss_p63 + pixelwise_cycle_loss_p63_inv

        # total cycle loss
        cycle_loss = (cycle_he_loss_total + cycle_p63_loss_total) * self.settings.lambda_cycle

        # identity loss
        identity_he = self.criterion_pixel_wise(real_he, self.generator_p63_to_he(real_he, mask_he))
        identity_p63 = self.criterion_pixel_wise(real_p63, self.generator_he_to_p63(real_p63, mask_p63))
        identity_loss = (identity_he + identity_p63) * self.settings.lambda_identity

        with torch.no_grad():
            discriminator_he_real_decision = self.discriminator_he(real_he)
            discriminator_he_real_loss = self.get_loss(discriminator_he_real_decision, self.criterion_GAN, torch.ones)
            discriminator_he_fake_decision = self.discriminator_he(fake_he)
            discriminator_he_fake_loss = self.get_loss(discriminator_he_fake_decision, self.criterion_GAN, torch.zeros)

            discriminator_p63_real_decision = self.discriminator_p63(real_p63)
            discriminator_p63_real_loss = self.get_loss(discriminator_p63_real_decision, self.criterion_GAN, torch.ones)
            discriminator_p63_fake_decision = self.discriminator_p63(fake_p63)
            discriminator_p63_fake_loss = self.get_loss(discriminator_p63_fake_decision, self.criterion_GAN,
                                                        torch.zeros)

            discriminator_he_mask_real_decision = self.discriminator_he_mask(real_he * mask_he)
            discriminator_he_mask_real_loss = self.get_loss(discriminator_he_mask_real_decision, self.criterion_GAN,
                                                            torch.ones)
            discriminator_he_mask_fake_decision = self.discriminator_he_mask(fake_he * mask_he)
            discriminator_he_mask_fake_loss = self.get_loss(discriminator_he_mask_fake_decision, self.criterion_GAN,
                                                            torch.zeros)

            discriminator_p63_mask_real_decision = self.discriminator_p63_mask(real_p63 * mask_p63)
            discriminator_p63_mask_real_loss = self.get_loss(discriminator_p63_mask_real_decision, self.criterion_GAN,
                                                             torch.ones)
            discriminator_p63_mask_fake_decision = self.discriminator_p63_mask(fake_p63 * mask_p63)
            discriminator_p63_mask_fake_loss = self.get_loss(discriminator_p63_mask_fake_decision, self.criterion_GAN,
                                                             torch.zeros)

            discriminator_he_loss_partial = (discriminator_he_real_loss + discriminator_he_fake_loss) * 0.5 * (
                        1 - self.settings.lambda_mask_adversarial_ratio)
            discriminator_he_loss_mask_partial = (
                discriminator_he_mask_real_loss + discriminator_he_mask_fake_loss
            ) * 0.5 * self.settings.lambda_mask_adversarial_ratio
            discriminator_p63_loss_partial = (discriminator_p63_real_loss + discriminator_p63_fake_loss) * 0.5 * (
                        1 - self.settings.lambda_mask_adversarial_ratio)
            discriminator_p63_loss_mask_partial = (
                discriminator_p63_mask_real_loss + discriminator_p63_mask_fake_loss
            ) * 0.5 * self.settings.lambda_mask_adversarial_ratio

            self.p63_explainer.set_losses(discriminator_he_loss_partial, discriminator_he_loss_mask_partial)
            self.he_explainer.set_losses(discriminator_p63_loss_partial, discriminator_p63_loss_mask_partial)
            self.p63_explainer.get_explanation()
            self.he_explainer.get_explanation()

        # backward gen
        generator_loss = generator_he_to_p63_total_loss + generator_p63_to_he_total_loss + cycle_loss + identity_loss
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()

        # Train discriminator self.discriminator_he
        discriminator_he_real_decision = self.discriminator_he(real_he)
        discriminator_he_real_loss = self.get_loss(discriminator_he_real_decision, self.criterion_GAN, torch.ones)
        fake_he = self.fake_he_pool.query(fake_he)
        discriminator_he_fake_decision = self.discriminator_he(fake_he)
        discriminator_he_fake_loss = self.get_loss(discriminator_he_fake_decision, self.criterion_GAN, torch.zeros)

        # Train discriminator self.discriminator_p63
        discriminator_p63_real_decision = self.discriminator_p63(real_p63)
        discriminator_p63_real_loss = self.get_loss(discriminator_p63_real_decision, self.criterion_GAN, torch.ones)
        fake_p63 = self.fake_p63_pool.query(fake_p63)
        discriminator_p63_fake_decision = self.discriminator_p63(fake_p63)
        discriminator_p63_fake_loss = self.get_loss(discriminator_p63_fake_decision, self.criterion_GAN, torch.zeros)

        # Train discriminator self.discriminator_he_mask
        discriminator_he_mask_real_decision = self.discriminator_he_mask(real_he * mask_he)
        discriminator_he_mask_real_loss = self.get_loss(discriminator_he_mask_real_decision, self.criterion_GAN,
                                                        torch.ones)
        fake_he = self.fake_he_pool.query(fake_he)
        discriminator_he_mask_fake_decision = self.discriminator_he_mask(fake_he * mask_he)
        discriminator_he_mask_fake_loss = self.get_loss(discriminator_he_mask_fake_decision, self.criterion_GAN,
                                                        torch.zeros)

        # Train discriminator self.discriminator_p63_mask
        discriminator_p63_mask_real_decision = self.discriminator_p63_mask(real_p63 * mask_p63)
        discriminator_p63_mask_real_loss = self.get_loss(discriminator_p63_mask_real_decision, self.criterion_GAN,
                                                         torch.ones)
        fake_p63 = self.fake_p63_pool.query(fake_p63)
        discriminator_p63_mask_fake_decision = self.discriminator_p63_mask(fake_p63 * mask_p63)
        discriminator_p63_mask_fake_loss = self.get_loss(discriminator_p63_mask_fake_decision, self.criterion_GAN,
                                                         torch.zeros)

        # Back propagation
        discriminator_he_loss_partial = (discriminator_he_real_loss + discriminator_he_fake_loss) * 0.5 * (
                    1 - self.settings.lambda_mask_adversarial_ratio)
        discriminator_he_loss_mask_partial = (
            discriminator_he_mask_real_loss + discriminator_he_mask_fake_loss
        ) * 0.5 * self.settings.lambda_mask_adversarial_ratio
        discriminator_he_loss = discriminator_he_loss_partial + discriminator_he_loss_mask_partial

        self.discriminator_he_optimizer.zero_grad()
        discriminator_he_loss.backward()
        self.discriminator_he_optimizer.step()

        discriminator_p63_loss_partial = (discriminator_p63_real_loss + discriminator_p63_fake_loss) * 0.5 * (
                    1 - self.settings.lambda_mask_adversarial_ratio)
        discriminator_p63_loss_mask_partial = (
            discriminator_p63_mask_real_loss + discriminator_p63_mask_fake_loss
        ) * 0.5 * self.settings.lambda_mask_adversarial_ratio
        discriminator_p63_loss = discriminator_p63_loss_partial + discriminator_p63_loss_mask_partial

        self.discriminator_p63_optimizer.zero_grad()
        discriminator_p63_loss.backward()
        self.discriminator_p63_optimizer.step()

        self.latest_generator_loss = generator_loss.item()
        self.latest_discriminator_he_loss = discriminator_he_loss.item()
        self.latest_discriminator_p63_loss = discriminator_p63_loss.item()

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

        real_he = Variable(real_he.to(self.device))
        real_p63 = Variable(real_p63.to(self.device))

        real_he_mask = get_mask(real_he, self.settings.mask_type)
        real_p63_mask = get_mask(real_p63, self.settings.mask_type)

        real_he_mask = Variable(real_he_mask.to(self.device))
        real_p63_mask = Variable(real_p63_mask.to(self.device))

        fake_p63 = self.generator_he_to_p63(real_he, real_he_mask)
        reconstructed_he = self.generator_p63_to_he(fake_p63, real_he_mask)

        fake_he = self.generator_p63_to_he(real_p63, real_p63_mask)
        reconstructed_p63 = self.generator_he_to_p63(fake_he, real_p63_mask)

        return (real_he, real_p63), (fake_he, fake_p63), (reconstructed_he, reconstructed_p63)
