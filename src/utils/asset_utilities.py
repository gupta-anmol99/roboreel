from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.io_utils import load_json

PARTNET_MOBILITY = None


def _load_partnet_mobility_dataset():
    global PARTNET_MOBILITY
    """loads preprocssed partnet mobility metadata"""
    from pathlib import Path
    configs_dir = ASSET_DIR / "configs"
    merged_data = {}
    for json_file in sorted(Path(configs_dir).glob("info_*.json")):
        merged_data.update(load_json(json_file))
    PARTNET_MOBILITY = {
        "model_data": merged_data,
    }
    # for data_file in ["info_cabinet_door_train.json", "info_faucet_train.json"]:
    #     PARTNET_MOBILITY["model_data"].update(
    #         load_json(PACKAGE_ASSET_DIR / "partnet_mobility/meta" / data_file)
    #     )

    def find_urdf_path(model_id):
        model_dir = ASSET_DIR / "dataset" / str(model_id)
        urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf", "mobility.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path

    PARTNET_MOBILITY["model_urdf_paths"] = {}
    for k in PARTNET_MOBILITY["model_data"].keys():
        urdf_path = find_urdf_path(k)
        if urdf_path is not None:
            PARTNET_MOBILITY["model_urdf_paths"][k] = urdf_path

    if len(PARTNET_MOBILITY["model_urdf_paths"]) == 0:
        raise RuntimeError(
            "Partnet Mobility dataset not found. Download it by running python -m mani_skill.utils.download_asset partnet_mobility_cabinet"
        )


def get_partnet_mobility_builder(
    scene: ManiSkillScene,
    id: str,
    fix_root_link: bool = True,
    urdf_config: dict = dict(),
):
    global PARTNET_MOBILITY
    if PARTNET_MOBILITY is None:
        _load_partnet_mobility_dataset()
    metadata = PARTNET_MOBILITY["model_data"][id]
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.scale = metadata["scale"]
    loader.load_multiple_collisions_from_file = True
    urdf_path = PARTNET_MOBILITY["model_urdf_paths"][id]
    applied_urdf_config = sapien_utils.parse_urdf_config(
        dict(
            material=dict(static_friction=1, dynamic_friction=1, restitution=0),
        )
    )
    applied_urdf_config.update(**urdf_config)
    sapien_utils.apply_urdf_config(loader, applied_urdf_config)
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]
    return builder

def get_articulation_builder(
    scene: ManiSkillScene,
    id: str,
    fix_root_link: bool = True,
    urdf_config: dict = dict(),
) -> ArticulationBuilder:
    """Builds an articulation or returns an articulation builder given an ID specifying which dataset/source and then the articulation ID

    Currently these IDs are hardcoded for a few datasets. The new Shapedex platform for hosting and managing all assets will be
    integrated in the future
    """
    splits = id.split(":")
    dataset_source = splits[0]
    articulation_id = ":".join(splits[1:])

    if dataset_source == "partnet-mobility":
        builder = get_partnet_mobility_builder(
            scene=scene,
            id=articulation_id,
            fix_root_link=fix_root_link,
            urdf_config=urdf_config,
        )
    else:
        raise RuntimeError(f"No dataset with id {dataset_source} was found")

    return builder
