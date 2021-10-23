import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


def get_accum_card_error(target, preds, num_cards=52):
    cards_target = target * num_cards
    cards_pred = preds * num_cards
    card_diff = abs(cards_pred - cards_target)
    return card_diff.sum()


# UNUSED
def card_loss(out_detections, out_coords):
    diff_x = out_coords[:, 8::2] - out_coords[:, :8:2]
    diff_y = out_coords[:, ::4] - out_coords[:, 2::4]
    zeros = torch.zeros_like(diff_x)
    loss_y = torch.sum(torch.abs(torch.min(zeros, diff_y)), dim=1)
    loss_x = torch.sum(torch.abs(torch.min(zeros, diff_x)), dim=1)
    loss = loss_x + loss_y
    return loss.mean()
