# http://idd.insaan.iiit.ac.in/dataset/details/

import glob
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "IDD segmentation"
    dataset_path = "/home/grokhi/rawdata/idd/"

    batch_size = 30
    images_folder = "leftImg8bit"
    ann_json_folder = "gtFine"
    ann_suffix = "_gtFine_polygons.json"

    def create_ann(image_path):
        global meta
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        seq_value = int(image_path.split("/")[-2])
        seq = sly.Tag(seq_folder, value=seq_value)

        if ds_name != "test":
            image_name = get_file_name_with_ext(image_path)
            im_path = image_path[: -1 * len(image_name)]
            ann_path = im_path.replace("leftImg8bit", "gtFine")

            ann_name = get_file_name(image_path).split("_leftImg8bit")[0]
            ann_name += ann_suffix

            ann_path += ann_name
            ann_data = load_json_file(ann_path)

            for curr_ann_data in ann_data["objects"]:
                class_name = curr_ann_data["label"]
                level4id, level3id, category, level2id, level1id = class_to_levels.get(
                    class_name, [-1, -1, "unspecified", -1, -1]
                )

                level4id, level3id, category, level2id, level1id = (
                    int(level4id),
                    int(level3id),
                    str(category),
                    int(level2id),
                    int(level1id),
                )

                vals = [category, level1id, level2id, level3id, level4id]
                label_tags = [
                    sly.Tag(tag_meta, value=val) for tag_meta, val in zip(tag_metas, vals)
                ]
                if category == "unspecified":
                    label_tags = [sly.Tag(tag_category, value="unspecified")]

                obj_class = meta.get_obj_class(class_name)
                if obj_class is None:
                    obj_class = sly.ObjClass(class_name, sly.Polygon)
                    meta = meta.add_obj_class(obj_class)
                    api.project.update_meta(project.id, meta)

                polygons_coords = curr_ann_data["polygon"]
                exterior = []
                for coords in polygons_coords:
                    exterior.append([int(coords[1]), int(coords[0])])

                if len(exterior) < 3:
                    continue
                poligon = sly.Polygon(exterior)
                label_poly = sly.Label(poligon, obj_class, tags=label_tags)
                labels.append(label_poly)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[seq])

    labels_ = [
        # name id csId csTrainId level4id level3Id category level2Id level1Id hasInstances ignoreInEval color
        ("road", 0, 7, 0, 0, 0, "drivable", 0, 0, False, False, (128, 64, 128)),
        ("parking", 1, 9, 255, 1, 1, "drivable", 1, 0, False, False, (250, 170, 160)),
        ("drivable fallback", 2, 255, 255, 2, 1, "drivable", 1, 0, False, False, (81, 0, 81)),
        ("sidewalk", 3, 8, 1, 3, 2, "non-drivable", 2, 1, False, False, (244, 35, 232)),
        (
            "non-drivable fallback",
            5,
            255,
            9,
            4,
            3,
            "non-drivable",
            3,
            1,
            False,
            False,
            (152, 251, 152),
        ),
        ("rail track", 4, 10, 255, 3, 3, "non-drivable", 3, 1, False, False, (230, 150, 140)),
        ("person", 6, 24, 11, 5, 4, "living-thing", 4, 2, True, False, (220, 20, 60)),
        ("animal", 7, 255, 255, 6, 4, "living-thing", 4, 2, True, True, (246, 198, 145)),
        ("rider", 8, 25, 12, 7, 5, "living-thing", 5, 2, True, False, (255, 0, 0)),
        ("motorcycle", 9, 32, 17, 8, 6, "2-wheeler", 6, 3, True, False, (0, 0, 230)),
        ("bicycle", 10, 33, 18, 9, 7, "2-wheeler", 6, 3, True, False, (119, 11, 32)),
        ("autorickshaw", 11, 255, 255, 10, 8, "autorickshaw", 7, 3, True, False, (255, 204, 54)),
        ("car", 12, 26, 13, 11, 9, "car", 7, 3, True, False, (0, 0, 142)),
        ("truck", 13, 27, 14, 12, 10, "large-vehicle", 8, 3, True, False, (0, 0, 70)),
        ("bus", 14, 28, 15, 13, 11, "large-vehicle", 8, 3, True, False, (0, 60, 100)),
        ("caravan", 15, 29, 255, 14, 12, "large-vehicle", 8, 3, True, True, (0, 0, 90)),
        ("trailer", 16, 30, 255, 15, 12, "large-vehicle", 8, 3, True, True, (0, 0, 110)),
        ("train", 17, 31, 16, 15, 12, "large-vehicle", 8, 3, True, True, (0, 80, 100)),
        (
            "vehicle fallback",
            18,
            355,
            255,
            15,
            12,
            "large-vehicle",
            8,
            3,
            True,
            False,
            (136, 143, 153),
        ),
        ("curb", 19, 255, 255, 16, 13, "barrier", 9, 4, False, False, (220, 190, 40)),
        ("wall", 20, 12, 3, 17, 14, "barrier", 9, 4, False, False, (102, 102, 156)),
        ("fence", 21, 13, 4, 18, 15, "barrier", 10, 4, False, False, (190, 153, 153)),
        ("guard rail", 22, 14, 255, 19, 16, "barrier", 10, 4, False, False, (180, 165, 180)),
        ("billboard", 23, 255, 255, 20, 17, "structures", 11, 4, False, False, (174, 64, 67)),
        ("traffic sign", 24, 20, 7, 21, 18, "structures", 11, 4, False, False, (220, 220, 0)),
        ("traffic light", 25, 19, 6, 22, 19, "structures", 11, 4, False, False, (250, 170, 30)),
        ("pole", 26, 17, 5, 23, 20, "structures", 12, 4, False, False, (153, 153, 153)),
        ("polegroup", 27, 18, 255, 23, 20, "structures", 12, 4, False, False, (153, 153, 153)),
        (
            "obs-str-bar-fallback",
            28,
            255,
            255,
            24,
            21,
            "structures",
            12,
            4,
            False,
            False,
            (169, 187, 214),
        ),
        ("building", 29, 11, 2, 25, 22, "construction", 13, 5, False, False, (70, 70, 70)),
        ("bridge", 30, 15, 255, 26, 23, "construction", 13, 5, False, False, (150, 100, 100)),
        ("tunnel", 31, 16, 255, 26, 23, "construction", 13, 5, False, False, (150, 120, 90)),
        ("vegetation", 32, 21, 8, 27, 24, "vegetation", 14, 5, False, False, (107, 142, 35)),
        ("sky", 33, 23, 10, 28, 25, "sky", 15, 6, False, False, (70, 130, 180)),
        (
            "fallback background",
            34,
            255,
            255,
            29,
            25,
            "object fallback",
            15,
            6,
            False,
            False,
            (169, 187, 214),
        ),
        ("unlabeled", 35, 0, 255, 255, 255, "void", 255, 255, False, True, (0, 0, 0)),
        ("ego vehicle", 36, 1, 255, 255, 255, "void", 255, 255, False, True, (0, 0, 0)),
        ("rectification border", 37, 2, 255, 255, 255, "void", 255, 255, False, True, (0, 0, 0)),
        ("out of roi", 38, 3, 255, 255, 255, "void", 255, 255, False, True, (0, 0, 0)),
        ("license plate", 39, 255, 255, 255, 255, "vehicle", 255, 255, False, True, (0, 0, 142)),
    ]

    tag_names = ["level1id", "level2id", "level3id", "level4id"]
    tag_category = sly.TagMeta("category", sly.TagValueType.ANY_STRING)

    tag_metas = [tag_category] + [
        sly.TagMeta(name, sly.TagValueType.ANY_NUMBER) for name in tag_names
    ]

    seq_folder = sly.TagMeta("sequence", sly.TagValueType.ANY_NUMBER)
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    global meta
    meta = sly.ProjectMeta(tag_metas=tag_metas + [seq_folder])

    class_to_levels = {}
    for label in labels_:
        class_to_levels[label[0]] = [label[4], label[5], label[6], label[7], label[8]]
        obj_class = sly.ObjClass(label[0], sly.Polygon, color=label[-1])
        meta = meta.add_obj_class(obj_class)

    api.project.update_meta(project.id, meta.to_json())

    train_images_pathes = glob.glob(dataset_path + "./*/leftImg8bit/train/*/*.*")
    val_images_pathes = glob.glob(dataset_path + "./*/leftImg8bit/val/*/*.*")
    test_images_pathes = glob.glob(dataset_path + "./*/leftImg8bit/test/*/*.*")

    ds_to_data = {
        "train": train_images_pathes,
        "val": val_images_pathes,
        "test": test_images_pathes,
    }

    for ds_name, images_pathes in ds_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            img_names_batch = [
                im_path.split("/")[-2] + "_" + get_file_name_with_ext(im_path)
                for im_path in img_pathes_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))
    return project
