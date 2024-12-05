"""
This script is based on code from the 'torchattacks' library (https://github.com/Harry24k/torchattacks).
Significant portions of this script have been adapted or modified from the original implementation.

torchattacks is licensed under the MIT License:

MIT License

Copyright (c) 2020 Harry Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified and adapted by Itamar Kapuza, 2024.
"""


import torch
import torch.nn as nn

from torchattacks.attack import Attack


class FGSM(Attack):
    """
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation.

    Shape:
        - images: (N, C, L) where N is number of batches, C is number of channels and L is length.
        - labels: (N) where each value y_i is such that 0 <= y_i <= number of labels.
        - output: (N, C, L).
    """

    def __init__(self, model, eps=0.06):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps * grad.sign()

        return adv_images


class BIM(Attack):
    """
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        steps (int): number of steps.

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: (N, C, L) where N is number of batches, C is number of channels and L is length.
        - labels: (N) where each value y_i is integer such that 0 <= y_i < number of labels.
        - output: (N, C, L).
    """

    def __init__(self, model, eps=0.06, alpha=0.06 / 10, steps=10):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = 10
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        adv_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(adv_images, labels)

        loss = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps).detach()
            adv_images = images + delta

        return adv_images


class MIFGSM(Attack):
    """
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        decay (float): momentum factor.
        steps (int): number of iterations.

    Shape:
        - images: (N, C, L) where N is number of batches, C is number of channels and L is length.
        - labels: (N) where each value y_i is such that 0 <= y_i <= number of labels.
        - output: (N, C, L).
    """

    def __init__(self, model, eps=0.06, alpha=0.06 / 10, steps=10, decay=0.9):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps).detach()
            adv_images = images + delta

        return adv_images


class PGD(Attack):
    """
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        steps (int): number of steps.
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: (N, C, L) where N is number of batches, C is number of channels and L is length.
        - labels: (N) where each value y_i is such that 0 <= y_i <= number of labels.
        - output: (N, C, L).
    """

    def __init__(self, model, eps, alpha, steps, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps,
                                                                            generator=torch.cuda.manual_seed(self.eps))

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps).detach()
            adv_images = images + delta

        return adv_images
