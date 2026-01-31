# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to recalibrate your device (robot or teleoperator).

left_leader = /dev/tty.usbmodem5A7A0187661

left_arm = /dev/tty.usbmodem5AB01580631

Example:

```shell
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A7A0187661 \
    --teleop.id=left_leader

lerobot-calibrate \
    --teleop.type=so101_follower \
    --teleop.port=/dev/tty.usbmodem5AB01580631 \
    --teleop.id=left_arm

python -m lerobot.calibrate \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58FA1024931 \
    --robot.id=white
    ```

Jenn's arms:
```shell
python -m lerobot.calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A4B0466901 \
    --robot.id=jenn_white
```
"""

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.utils import init_logging


@dataclass
class CalibrateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)

    device.connect(calibrate=False)
    device.calibrate()
    device.disconnect()


def main():
    calibrate()


if __name__ == "__main__":
    main()
