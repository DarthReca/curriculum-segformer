# Copyright 2018- The Hugging Face team. All rights reserved.

import logging
from collections import deque
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, OrderedDict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from hydra.utils import instantiate
from lr_schedulers import CustomDecreasing
from pytorch_lightning import LightningModule
from torch import optim
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import JaccardIndex
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks, make_grid

from .segformer_backbone import (
    BaseModelOutput,
    CurriculumSegformerEncoder,
    SegformerEncoder,
)
from .segformer_head import CurriculumSegFormerHead, SegformerHead


class SemanticSegmenterOutput(NamedTuple):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# MODIFICATIONS OF THE MODEL TO WORK WITH PYTORCH LIGHTNING
class Segformer(LightningModule):
    def __init__(
        self,
        decode_head_params: Dict[str, Any],
        backbone_params: Dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = (
            CurriculumSegformerEncoder(**backbone_params)
            if self.hparams["curriculum"]["active"]
            else SegformerEncoder(**backbone_params)
        )
        self.decode_head = (
            CurriculumSegFormerHead(**decode_head_params)
            if self.hparams["curriculum"]["active"]
            else SegformerHead(**decode_head_params)
        )
        self.use_return_dict = True
        self.output_hidden_states = True
        self.num_labels = decode_head_params["num_classes"]
        self.semantic_loss_ignore_index = 255

        self.val_metrics = JaccardIndex(
            task="multiclass",
            num_classes=self.num_labels,
            ignore_index=self.semantic_loss_ignore_index,
            compute_on_cpu=True,
        )
        self.test_metrics = JaccardIndex(
            task="multiclass",
            num_classes=self.num_labels,
            ignore_index=self.semantic_loss_ignore_index,
        )

        self.indexes = torch.tensor(list(range(5)))

        self.selected_classes = self.hparams["selected_classes"]
        if self.selected_classes is None:
            self.selected_classes = set(
                [c.train_id for c in Cityscapes.classes if c.id != -1]
            )
        elif self.selected_classes == "all":
            self.selected_classes = list(
                [c.id for c in Cityscapes.classes if c.id != -1]
            )
        logging.info(f"Selected classes: {len(self.selected_classes)}")

        self._last_losses = deque(maxlen=self.hparams["curriculum"]["loss_window_size"])

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        resize_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:
        ```"""

        return_dict = return_dict if return_dict is not None else self.use_return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.output_hidden_states
        )

        outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
            output_next_layer=False,
        )

        logits = self.decode_head(outputs)
        contrastive_logits = None
        if isinstance(logits, tuple):
            logits, contrastive_logits = logits

        if resize_shape is None:
            resize_shape = labels.shape[-2:]

        upsampled_logits = nn.functional.interpolate(
            logits, size=resize_shape, mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            # upsample logits to the images' original size
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.semantic_loss_ignore_index)
            loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=upsampled_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def configure_optimizers(self) -> Any:
        adamw = optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        # Add total steps for OneCycleLR
        if "total_steps" in self.hparams["lr_scheduler"].keys():
            self.hparams["lr_scheduler"][
                "total_steps"
            ] = self.trainer.estimated_stepping_batches

        scheduler_params = dict(self.hparams["lr_scheduler"])
        lr_scheduler = instantiate(scheduler_params, optimizer=adamw)
        interval = (
            "step"
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR)
            else "epoch"
        )

        return {
            "optimizer": adamw,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": interval},
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Support checkpointing of CyclicScheduler
        if (
            "lr_schedulers" in checkpoint
            and len(checkpoint["lr_schedulers"]) > 0
            and "_scale_fn_ref" in checkpoint["lr_schedulers"][0]
        ):
            del checkpoint["lr_schedulers"][0]["_scale_fn_ref"]
            del checkpoint["lr_schedulers"][0]["_scale_fn_custom"]

    def training_step(self, batch, batch_idx):
        # Apply curriculum learning at the beginning of the step
        if (
            self.hparams["curriculum"]["insertion_unit"] == "step"
            and self.global_step in self.hparams["curriculum"]["insertion_interval"]
        ):
            self._apply_model_curriculum()

        img, mask = batch[0].float(), batch[1].squeeze()
        # Convert to 19 classes if needed
        if len(self.selected_classes) == 20:
            mask = self._convert_to_19_classes(mask)

        loss = self(img, mask).loss

        self.log("train_loss", loss)
        self._last_losses.append(loss.detach().cpu().numpy())

        return loss

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Apply curriculum learning at the end of the epoch
        if (
            self.hparams["curriculum"]["insertion_unit"] == "epoch"
            and self.current_epoch in self.hparams["curriculum"]["insertion_interval"]
        ):
            self._apply_model_curriculum()

        self.trainer.datamodule.current_epoch = self.current_epoch

    def validation_step(self, batch, batch_idx):
        img, mask = batch[0].float(), batch[1].squeeze()
        # Convert to 19 classes if needed
        if len(self.selected_classes) == 20:
            mask = self._convert_to_19_classes(mask)

        output = self(img, mask)
        cpu_logits = output.logits.detach().cpu()
        cpu_mask = mask.detach().cpu()
        self.val_metrics.to(torch.device("cpu"))(cpu_logits, cpu_mask)

        self.log("val_loss", output.loss)
        self.log("val_IoU", self.val_metrics)

        return {
            "val_loss": output.loss,
        }

    def test_step(self, batch, batch_idx):
        img, mask = batch[0].float(), batch[1].squeeze()
        # Convert to 19 classes if needed
        if len(self.selected_classes) == 20:
            mask = self._convert_to_19_classes(mask)

        selected_classes = torch.tensor(self.selected_classes, device=mask.device)
        mask = torch.isin(mask, selected_classes).long()

        output = self(img, resize_shape=mask.shape[-2:])
        self.test_metrics(output.logits, mask)
        self.log("test_IoU", self.test_metrics)
        return {"img": img, "gt": mask, "pred": output.logits}

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self.indexes = torch.tensor(list(range(10)))
        self._log_images(outputs[:2])

    def _log_images(
        self, step_output: List[Dict[str, torch.Tensor]], max_logged_images: int = 50
    ):
        merged = {
            k: torch.cat([d[k].cpu() for d in step_output])
            for k in ("img", "gt", "pred")
        }
        # Select some samples
        if self.indexes is None:
            self.indexes = torch.randint(
                low=0, high=merged["img"].shape[0], size=(max_logged_images,)
            )
        merged = {k: v[self.indexes] for k, v in merged.items()}
        # Select colors
        colors = [x.color for x in Cityscapes.classes if x.id in self.selected_classes]
        for i, (img, gt, pred) in enumerate(
            zip(merged["img"], merged["gt"], merged["pred"])
        ):
            # Draw masks and log
            gt = torch.stack([gt == id for id in self.selected_classes])
            pred = torch.stack([pred == id for id in self.selected_classes])
            img_gt = draw_segmentation_masks(img.byte(), gt.bool(), colors=colors)
            img_pred = draw_segmentation_masks(img.byte(), pred.bool(), colors=colors)
            self.logger.experiment.log_image(
                to_pil_image(make_grid([img_gt, img_pred], nrow=2)),
                name=f"{i}",
                step=self.global_step,
            )

    def _apply_model_curriculum(self):
        if (
            self.hparams["curriculum"]["active"]
            and not self.encoder.active_blocks.all()
        ):
            logging.log(logging.INFO, "INFO: Activating a new layer")
            self.encoder.activate_next_block()
            self.decode_head.activate_next_block()
            self._activated_next_layer = True

    def _convert_to_19_classes(self, mask):
        mask = mask.long()
        for c in Cityscapes.classes:
            mask[mask == c.id] = c.train_id
        return mask


id2label = (
    {
        "0": "road",
        "1": "sidewalk",
        "2": "building",
        "3": "wall",
        "4": "fence",
        "5": "pole",
        "6": "traffic light",
        "7": "traffic sign",
        "8": "vegetation",
        "9": "terrain",
        "10": "sky",
        "11": "person",
        "12": "rider",
        "13": "car",
        "14": "truck",
        "15": "bus",
        "16": "train",
        "17": "motorcycle",
        "18": "bicycle",
    },
)
