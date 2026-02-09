from functools import partial

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import  EvalPrediction, CLIPModel, CLIPProcessor


def compute_glue_metrics(eval_pred: EvalPrediction,
                         metric):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=preds, references=labels)
    return result


def compute_image_cls_metrics(eval_pred: EvalPrediction,
                              metric):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def _get_clip_model_and_processor(model_name: str):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name,
                                              use_fast=True,)
    return model, processor


def _pil_list_to_uint8_nchw(pil_images):
    arr = [np.asarray(im.convert("RGB"), dtype=np.uint8) for im in pil_images]  # list of [H,W,3], uint8
    t = torch.from_numpy(np.stack(arr, axis=0))                                 # [N,H,W,3]
    return t.permute(0, 3, 1, 2).contiguous()                                   # [N,3,H,W]


class SDText2ImageEvaluator:
    def __init__(self,
                 test_set,
                 image_transform,
                 device,
                 sample_seed: int,
                 clip_model_name_or_path: str,
                 caption_columns: str='text',
                 image_column: str='image',
    ):
        self.device = device
        self.test_set_size = len(test_set)

        self.prompts = [item[caption_columns] for item in test_set]

        real_pils = [item[image_column] for item in test_set]
        real_pils = [image_transform(im.convert("RGB")) for im in real_pils]
        real_uint8 = _pil_list_to_uint8_nchw(real_pils)

        self.sample_seeds = [int(sample_seed + 1234 + j) for j in range(self.test_set_size)]

        # Metrics
        self.fid = FrechetInceptionDistance(feature=2048,
                                            reset_real_features=False,
                                            normalize=False,).to(device)
        self.fid.set_dtype(torch.float64)

        model_name_or_path = partial(_get_clip_model_and_processor,
                                     model_name=clip_model_name_or_path)
        self.clip = CLIPScore(model_name_or_path=model_name_or_path).to(device)

        # Cache real features once
        with torch.inference_mode():
            self.fid.update(real_uint8.to(device, non_blocking=True), real=True)

    @torch.inference_mode()
    def evaluate(self,
                 pipe,
                 batch_size: int,
                 num_inference_steps: int,
                 guidance_scale: float,
                 num_images_per_prompt: int,
    ):
        self.fid.reset()
        self.clip.reset()

        pipe.unet.eval()

        for rep in range(num_images_per_prompt):
            seed_offset = rep * 1000
            for start in range(0, self.test_set_size, batch_size):
                end = min(start + batch_size, self.test_set_size)
                prompts_b = self.prompts[start:end]
                seeds_b = self.sample_seeds[start:end]
                gens = [torch.Generator(device=self.device).manual_seed(s + seed_offset) for s in seeds_b]

                with torch.autocast("cuda", dtype=pipe.unet.dtype):
                    out = pipe(prompts_b,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale,
                               generator=gens,)
                fake_uint8 = _pil_list_to_uint8_nchw(out.images).to(self.device, non_blocking=True)

                self.fid.update(fake_uint8, real=False)
                self.clip.update(fake_uint8, prompts_b)

        fid_val = float(self.fid.compute().detach().cpu())
        clip_val = float(self.clip.compute().detach().cpu())
        return {
            "fid": fid_val,
            "clip_score": clip_val,
            "num_generated": self.test_set_size * num_images_per_prompt,
        }
