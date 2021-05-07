import ast
import configparser
import logging
import os.path
import subprocess
import tempfile
from typing import Dict

import cv2
import numpy as np
from typing_extensions import Literal

logger = logging.getLogger("texsymdetect")

Path = str
PageNumber = int


class DocumentRasterException(Exception):
    pass


"""
Load commands for rastering TeX outputs.
"""
RASTER_CONFIG = "config.ini"
config = configparser.ConfigParser()
config.read(RASTER_CONFIG)


def raster_pages(
    document: Path, document_type: Literal["ps", "pdf"],
) -> Dict[PageNumber, np.array]:
    raster_commands: Dict[str, str] = {}
    if "rasterers" in config:
        raster_commands = dict(config["rasterers"])

    try:
        raster_command = raster_commands[document_type]
    except KeyError:
        raise DocumentRasterException(
            f"Could not find a rastering command for file {document} "
            + f"of type {document_type}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        args = ast.literal_eval(raster_command)
        args_resolved = [arg.format(output_dir=temp_dir, file=document) for arg in args]

        logger.debug("Attempting to raster pages for file %s.", document)
        result = subprocess.run(
            args_resolved, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode == 0:
            logger.debug(
                "Successfully rastered pages for file %s using command %s.",
                document,
                args_resolved,
            )
        else:
            raise DocumentRasterException(
                f"Error rastering file {document} using command {str(args_resolved)}: "
                + f"(Stdout: {result.stdout.decode(errors='ignore')}), "
                + f"(Stderr: {result.stderr.decode(errors='ignore')}).",
            )

        page_images: Dict[PageNumber, np.array] = {}
        for filename in os.listdir(temp_dir):
            page_no = int(filename.split(".")[0][5:])
            image = cv2.imread(os.path.join(temp_dir, filename))
            page_images[page_no] = image

        if not page_images:
            raise DocumentRasterException(
                f"No images of pages were generated when rastering {document}"
                + "despite a successful raster command."
            )

    return page_images
