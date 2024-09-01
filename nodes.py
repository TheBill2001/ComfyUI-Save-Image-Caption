import os
import glob
import hashlib
import json

from PIL import Image, ImageSequence, ImageOps
from PIL.PngImagePlugin import PngInfo
import torch
import numpy as np

import node_helpers
import folder_paths
from comfy.cli_args import args
from server import PromptServer


class BaseNode:
    NODE_CLASS_MAPPING = "CFICLS_"
    NODE_DISPLAY_NAME_MAPPING: str

    CATEGORY = "image/batch"

    DESCRIPTION: str = str()
    OUTPUT_TOOLTIPS: tuple[str] = ()

    @classmethod
    def INPUT_TYPES(s):
        return dict()


class LoadImageBatch(BaseNode):
    NODE_CLASS_MAPPING = BaseNode.NODE_CLASS_MAPPING + "LoadImageBatch"
    NODE_DISPLAY_NAME_MAPPING = "Batch Load Image"

    OUTPUT_TOOLTIPS = ("Batch loading images from a path.",)
    DESCRIPTION = "Batch loading images from a path."

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")

    OUTPUT_IS_LIST = (True, True)

    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING",),
                "recursive": (
                    "BOOLEAN",
                    {
                        "label_on": "yes",
                        "label_off": "no",
                        "default": False,
                        "defaultInput": False,
                    },
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(s, path: str, recursive: bool = False):
        m = hashlib.sha256(path.encode("utf-8"))
        for image in s.list_images(path, recursive):
            with open(image, "rb") as f:
                m.update(f.read())
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(s, path: str):
        if os.path.exists(path):
            return True
        return f'"{path}" does not exist'

    @classmethod
    def list_images(s, path: str, recursive: bool = False):
        images: list[str] = []
        pattern = "**/**" if recursive else "*"

        if os.path.isfile(path):
            files = [path]
        else:
            files = sorted(glob.glob(os.path.join(path, pattern), recursive=recursive))

        for filename in files:
            if os.path.isdir(filename):
                continue
            try:
                with Image.open(filename) as img:
                    img.verify()
                    images.append(filename)
            except (IOError, SyntaxError):
                pass
        return images

    def load(self, path: str, recursive: bool = False, node_id: str = None):
        images = []
        masks = []
        filepaths = self.list_images(path, recursive)

        for index, image_path in enumerate(filepaths):
            img = node_helpers.pillow(Image.open, image_path)

            output_images = []
            output_masks = []
            w, h = None, None

            excluded_formats = ["MPO"]

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == "I":
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]

                if image.size[0] != w or image.size[1] != h:
                    continue

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if "A" in i.getbands():
                    mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            images.append(output_image)
            masks.append(output_mask)

            PromptServer.instance.send_sync(
                "progress", {"node": node_id, "max": len(filepaths), "value": index}
            )

        return (images, masks)


class SaveImageCaptionBatch(BaseNode):
    NODE_CLASS_MAPPING = BaseNode.NODE_CLASS_MAPPING + "SaveImageCaptionBatch"
    NODE_DISPLAY_NAME_MAPPING = "Batch Save Image and Caption"

    OUTPUT_TOOLTIPS = ("Batch saving images and captions from a path.",)
    DESCRIPTION = "Batch saving images and captions from a path."

    INPUT_IS_LIST = True

    FUNCTION = "save"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "captions": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "path": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": "IMG"}),
                "extension": ("STRING", {"default": ".txt"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "node_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, images: list, captions: list):
        if not len(images) == len(captions):
            return "Images and captions size mitmatch!"
        return True

    def save(
        self,
        images: list,
        captions: list[str],
        prefix: list[str] = ["IMG"],
        extension: list[str] = [".txt"],
        path: list[str] = [""],
        prompt=[],
        extra_pnginfo=[],
        node_id=None,
    ):
        if len(prefix) == 0:
            prefix = "IMG"
        else:
            prefix = prefix[0]

        if len(extension) == 0:
            extension = ".txt"
        elif not extension[0].startswith("."):
            extension = "." + extension[0]
        else:
            extension = extension[0]
        if len(path) == 0 or path[0] == "":
            output_folder = folder_paths.get_output_directory()
        else:
            output_folder = path[0]
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
        if len(prompt) > 0:
            prompt = prompt[0]
        else:
            prompt = None

        if len(extra_pnginfo) > 0:
            extra_pnginfo = extra_pnginfo[0]
        else:
            extra_pnginfo = None
        for index, (image, caption) in enumerate(zip(images, captions)):
            filename = f"{prefix}_{index:05}"
            save_path = os.path.join(output_folder, filename)
            for batch_number, image in enumerate(image):
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                img.save(f"{save_path}_{batch_number}.png", pnginfo=metadata, compress_level=4)

            with open(f"{save_path}{extension}", "w", encoding="utf-8") as f:
                f.write(caption)
            PromptServer.instance.send_sync(
                "progress", {"node": node_id, "max": len(images), "value": index}
            )

        return ()
