import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import turtle_policy


@dataclasses.dataclass(frozen=True)
class TurtleDataConfig(DataConfigFactory):
    """Data pipeline config for Turtle2 realworld env with pi0/pi0.5.

    Turtle env provides:
      - observation/image            (main camera, 224x224x3)
      - observation/extra_view_image (2 extra cameras stacked, 224x2x224x3)
      - observation/state            (6-dim: xyz + euler, single arm)
      - actions                      (6-dim: xyz_delta + rpy_delta)
      - prompt                       (task description string)
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/extra_view_image": "observation/extra_view_image",
                        "observation/state": "observation/state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                turtle_policy.TurtleInputs(model_type=model_config.model_type)
            ],
            outputs=[turtle_policy.TurtleOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
