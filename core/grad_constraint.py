import numpy as np
import torch
from torchvision import transforms


class HookModule:
    def __init__(self, model, module):
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        self.model.zero_grad()
        return grads


class GradConstraint:

    def __init__(self, model, modules, channel_paths):
        print('- Grad Constraint')
        self.modules = []
        self.channels = []

        for module in modules:
            self.modules.append(HookModule(model=model, module=module))
        for channel_path in channel_paths:
            self.channels.append(torch.from_numpy(np.load(channel_path)).cuda())

    def loss_channel(self, outputs, labels):
        # high response channel loss
        probs = torch.argsort(-outputs, dim=1)
        labels_ = []
        for i in range(labels.size(0)):
            if probs[i][0] == labels[i]:
                labels_.append(probs[i][1])  # TP rank2
            else:
                labels_.append(probs[i][0])  # FP rank1
        labels_ = torch.tensor(labels_).cuda()
        nll_loss_ = torch.nn.NLLLoss()(outputs, labels_)
        # low response channel loss
        nll_loss = torch.nn.NLLLoss()(outputs, labels)

        loss = 0
        for i, module in enumerate(self.modules):
            # high response channel loss
            loss += _loss_channel(channels=self.channels[i],
                                  grads=module.grads(outputs=-nll_loss_,
                                                     inputs=module.activations),
                                  labels=labels_,
                                  is_high=True)

            # low response channel loss
            loss += _loss_channel(channels=self.channels[i],
                                  grads=module.grads(outputs=-nll_loss,
                                                     inputs=module.activations),
                                  labels=labels,
                                  is_high=False)
        return loss

    def loss_spatial(self, outputs, labels, masks):
        nll_loss = torch.nn.NLLLoss()(outputs, labels)
        grads = self.modules[0].grads(outputs=-nll_loss,
                                      inputs=self.modules[0].activations)
        masks = transforms.Resize((grads.shape[2], grads.shape[3]))(masks)
        masks_bg = 1 - masks
        grads_bg = torch.abs(masks_bg * grads)

        loss = grads_bg.sum()
        return loss


def _loss_channel(channels, grads, labels, is_high=True):
    grads = torch.abs(grads)
    channel_grads = torch.sum(grads, dim=(2, 3))  # [batch_size, channels]

    loss = 0
    if is_high:
        for b, l in enumerate(labels):
            loss += (channel_grads[b] * channels[l]).sum()
    else:
        for b, l in enumerate(labels):
            loss += (channel_grads[b] * (1 - channels[l])).sum()
    loss = loss / labels.size(0)
    return loss

# AC
# loss = torch.abs(grads).sum() / labels_neg.size(0)
# -CHC
# loss = 0
# grads = torch.abs(grads)
# channel_grads = torch.sum(grads, dim=(2, 3))  # [batch_size, channels]
# for b in range(len(labels)):
#     channels = torch.where((self.channels[labels_neg[b]]
#                             - self.channels[labels[b]]) > 0, 1, 0)
#     loss += (channel_grads[b] * channels).sum()
# loss = loss / labels.size(0)
