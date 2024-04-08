from typing import List

import cv2 as cv
import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    LlavaForConditionalGeneration,
    AutoProcessor,
)
from tqdm.notebook import tqdm
from gmflow.gmflow.gmflow import GMFlow
from Pbar import ProgressBar


class CreateCaption:
    """Class to create captions for videos using image analysis and text generation."""

    def __init__(
        self,
        device: torch.device,
        gm_loc: str,
        model_type="Salesforce/blip2-flan-t5-xxl",
        frames_to_skip=15,
        batch_size=64,
        avg_velocity=60,
    ) -> None:
        """
        Initialize CreateCaption class.

        Args:
            device (torch.device): Device to run the models on.
            gm_loc (str): Path to the model checkpoint for GMFlow.
            model_type (str, optional): Type of Blip2 model. Defaults to "Salesforce/blip2-flan-t5-xxl".
        """
        self.device = device
        self.model_type = model_type
        self.model_tools = {
            "llava": [AutoProcessor, LlavaForConditionalGeneration],
            "blip": [Blip2Processor, Blip2ForConditionalGeneration],
        }
        self.load_models(gm_loc)
        self.vel = avg_velocity
        self.batch_size = batch_size
        self.frames_to_skip = frames_to_skip
        self.prompt = """Caption this image:
Use below given text in square brackets
[{OCR}]
which are text on image in no perticular order
generate good caption describing entire image with text"""
        self.prompt = (
            "<image>" + self.prompt if "llava" in self.model_type else self.prompt
        )
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def load_models(self, gm_loc: str) -> None:
        """
        Load models required for caption generation.

        Args:
            gm_loc (str): Path to the model checkpoint for GMFlow.
        """
        self.model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(self.device)
        print("Loading GMFLow")
        checkpoint = torch.load(gm_loc, map_location=self.device)
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(weights, strict="store_true")
        print("Loading VLM")
        self.model_blip = self.model_tools[
            "llava" if "llava" in self.model_type else "blip"
        ][1].from_pretrained(self.model_type, load_in_4bit=True, device_map="auto")
        self.vis_processors = self.model_tools[
            "llava" if "llava" in self.model_type else "blip"
        ][0].from_pretrained(self.model_type)
        print("Loaded Models")

    def captions(self, video_loc: str, pr=1, total=1):
        """
        Generate captions for a given video.

        Args:
            video_loc (str): Path to the video file.
            pr (int, optional): Progress indicator. Defaults to 1.
            total (int, optional): Total number of videos. Defaults to 1.

        Returns:
            List[List[str, float]]: Captions with corresponding timestamps.
        """
        video = cv.VideoCapture(video_loc)
        prev_frames, next_frames, framesForCaptions, times, goodframes = (
            [],
            [],
            [],
            [],
            [],
        )
        pb = ProgressBar(1)
        fps = video.get(cv.CAP_PROP_FPS)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        c = 0
        count = 0
        time = 0
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        while True:
            for i in range(self.frames_to_skip):
                _, frame1 = video.read()
            if not _:
                break
            for i in range(self.frames_to_skip):
                _, frame2 = video.read()
            if not _:
                break
            pb.print(
                c * 2 * self.frames_to_skip,
                total_frames,
                f""" 
            \r Total frames read {c*10}/{total_frames} 
            \r time = {time}
            \r good frames ={len(goodframes)}
            \r {pr}/{total} videos""",
            )
            c += 1
            time = c * 2 * self.frames_to_skip / fps
            framesForCaptions.append(frame2)
            prev_frames.append(
                torch.from_numpy(
                    np.flip(cv.resize(frame1, (320, 160)), axis=-1).copy()
                ).permute(2, 0, 1)
            )
            next_frames.append(
                torch.from_numpy(
                    np.flip(cv.resize(frame2, (320, 160)), axis=-1).copy()
                ).permute(2, 0, 1)
            )
            times.append(time)
            count += 1
            with torch.no_grad():
                if count == self.batch_size:
                    image1 = torch.stack(prev_frames).float()
                    image2 = torch.stack(next_frames).float()
                    results_dict = self.model(
                        image1.to(self.device),
                        image2.to(self.device),
                        attn_splits_list=[2],
                        corr_radius_list=[-1],
                        prop_radius_list=[-1],
                        pred_bidir_flow=False,
                    )
                    velocity = results_dict["flow_preds"][-1]
                    velocity = torch.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
                    velocity = (
                        torch.sum(velocity, axis=[1, 2]) / sum(velocity.shape[1:])
                        > self.vel
                    )
                    for ind in tqdm(range(len(velocity))):
                        if velocity[ind]:
                            result = self.ocr.ocr(
                                np.flip(framesForCaptions[ind], axis=-1), cls=True
                            )
                            frame_0 = Image.fromarray(
                                np.flip(framesForCaptions[ind], axis=-1)
                            )
                            prompt = self.prompt.format(
                                OCR=" , ".join([x[1][0] for x in result[0]])
                            )
                            image = self.vis_processors(
                                images=frame_0, text=prompt, return_tensors="pt"
                            ).to(self.device)
                            out = self.model_blip.generate(**image, max_new_tokens=80)
                            caption = self.vis_processors.decode(
                                out[0], skip_special_tokens=True
                            )
                            goodframes.append(
                                [
                                    caption,
                                    times[ind],
                                ]
                            )
                    count = 0
                    prev_frames, next_frames, framesForCaptions, times = [], [], [], []
        return goodframes
