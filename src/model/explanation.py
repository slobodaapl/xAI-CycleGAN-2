from __future__ import annotations

from enum import Enum
from typing import Callable

from captum.attr import Saliency
from torch import cat, ones


# available explanation ramps based on name. The value is the exponent applied to the lambda function
class Ramp(Enum):
    linear = 1
    fast_start = 0.5
    slow_start = 2


class ExplanationController:

    def __init__(
            self,
            discriminator: Callable,
            discriminator_mask: Callable,
            mask_lambda: float = 0.7,
            ramp="fast_start"
    ):

        """
        Handles producing explanations for the generator.

        :param discriminator: The discriminator model of the GAN-in-training.
        :param discriminator_mask: The mask discriminator model of the GAN-in-training.
        :param mask_lambda: The weight of the mask discriminator loss.
        :param ramp: Determines the steepness of the loss ramp.
        """

        self.discriminator = discriminator
        self.discriminator_mask = discriminator_mask

        self.discriminator_loss = None
        self.discriminator_mask_loss = None

        self.explanation = ones(2, 3, 128, 128)
        self.explanation_mask = ones(2, 3, 128, 128)

        self.explainer = Saliency(self.discriminator)
        self.explainer_m = Saliency(self.discriminator_mask)

        self.loss_coefficient_function = lambda x: ((0.5 - min(0.5, x)) / 0.5) if x >= 0 else 0
        self.slope_coefficient = Ramp[ramp].value
        self.mask_lambda = mask_lambda
        self.explain_map = None

    def set_explanation(self, generated_data):
        exp = [self.explainer.attribute(
            generated_data[i, :].detach().unsqueeze(0), abs=False
        ) for i in range(generated_data.size(0))]

        self.explanation = cat(exp, dim=0)

    def set_explanation_m(self, generated_data):
        exp = [self.explainer_m.attribute(
            generated_data[i, :].detach().unsqueeze(0), abs=False
        ) for i in range(generated_data.size(0))]

        self.explanation_mask = cat(exp, dim=0)

    def get_coefficient_mask(self, loss):
        return self.loss_coefficient_function(loss) ** self.slope_coefficient

    # generate the explanation map based on the current losses, as described in the paper
    def get_explanation(self):
        self.explain_map = self.explanation * (1 - self.mask_lambda) \
                           * self.loss_coefficient_function(self.discriminator_loss) ** self.slope_coefficient \
                           + self.explanation_mask * self.mask_lambda \
                           * self.loss_coefficient_function(self.discriminator_mask_loss) ** self.slope_coefficient
        self.explain_map /= 2
        return self.explain_map

    def set_losses(self, loss_disc, loss_disc_m):
        self.discriminator_loss = loss_disc.detach()
        self.discriminator_mask_loss = loss_disc_m.detach()

    def set_losses_raw(self, loss_disc, loss_disc_m):
        self.discriminator_loss = loss_disc
        self.discriminator_mask_loss = loss_disc_m

    # add a hook to the backprop pass, where the explanation map is applied to the gradient in the last layer of the gen
    def explanation_hook(self, module, grad_input, grad_output):
        del module, grad_output
        out = grad_input[0] + grad_input[0] * self.explain_map
        return out.type(grad_input[0].type()),
