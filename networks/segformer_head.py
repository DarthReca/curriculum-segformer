# Copyright 2018- The Hugging Face team. All rights reserved.
# MODIFICATIONS TO MADE THE MODEL COMPATIBLE WITH PYTORCH

import logging
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, hidden_size, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerHead(nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int,
        classifier_dropout_prob: float,
        num_classes: int,
        reshape_last_stage: bool,
        **kwargs,
    ):
        super().__init__()
        num_encoder_blocks = len(hidden_sizes)
        self.num_encoder_blocks = num_encoder_blocks
        self.reshape_last_stage = reshape_last_stage
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(num_encoder_blocks):
            mlp = SegformerMLP(decoder_hidden_size, input_dim=hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Conv2d(decoder_hidden_size, num_classes, kernel_size=1)

    def forward(self, mit_output, **kwargs):
        encoder_hidden_states = mit_output.hidden_states
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = []
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            # upsample
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += [encoder_hidden_state]
        if hasattr(mit_output, "normalized_time") and mit_output.normalized_time != 1:
            all_hidden_states[-1] = torch.lerp(
                all_hidden_states[-2], all_hidden_states[-1], mit_output.normalized_time
            )
        # Duplicate oldest channel
        while len(all_hidden_states) != self.num_encoder_blocks:
            all_hidden_states += (all_hidden_states[0],)
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


"""
Curriculum Version
"""


class CurriculumSegFormerHead(SegformerHead):
    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int,
        classifier_dropout_prob: float,
        num_classes: int,
        reshape_last_stage: bool,
        insertion_time: int,
    ):
        super().__init__(
            hidden_sizes,
            decoder_hidden_size,
            classifier_dropout_prob,
            num_classes,
            reshape_last_stage,
        )
        num_encoder_blocks = len(hidden_sizes)

        self.active_blocks = torch.zeros(num_encoder_blocks, dtype=torch.bool)
        self.active_blocks[0] = True
        # Track time for smooth insertion
        self.insertion_time = insertion_time
        self._current_time = insertion_time

        self.linear_fuse = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=decoder_hidden_size * i,
                    out_channels=decoder_hidden_size,
                    kernel_size=1,
                    bias=False,
                )
                for i in range(1, num_encoder_blocks + 1)
            ]
        )

    def activate_next_block(self):
        if self.active_blocks.all():
            return
        self.active_blocks[(self.active_blocks == False).nonzero()[0]] = True
        self._current_time = 0

    def forward(self, mit_output, **kwargs):
        # Update time passing
        self._current_time = min(self.insertion_time, self._current_time + 1)
        normalized_time = self._current_time / self.insertion_time
        logging.getLogger("debug").debug(f"Normalized time: {normalized_time}")
        # SegFormer Forward
        encoder_hidden_states = mit_output.hidden_states
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = []
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )
            # upsample
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += [encoder_hidden_state]

        # Linear fuse with convolution
        num_active_blocks = self.active_blocks.sum()
        hidden_states = self.linear_fuse[len(all_hidden_states) - 1](
            torch.cat(all_hidden_states[::-1], dim=1)
        )
        # Smooth insertion
        if normalized_time != 1:
            stable_hs = self.linear_fuse[num_active_blocks - 2](
                torch.cat(all_hidden_states[:-1][::-1], dim=1)
            )
            hidden_states = torch.lerp(stable_hs, hidden_states, normalized_time)

        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits
