import os
import numpy as np
from utils.terrain_tool.terrain_generator import TerrainGenerator


def obstacle_corridor(tg: TerrainGenerator) -> None:
    BASE_HEIGHT = -0.5
    FLOOR_WIDTH = 5.0
    COURSE_LENGTH = 1000.0
    OBSTACLES = 100
    # floor
    tg.AddBox(
        position=[0.0, 0.0, BASE_HEIGHT],
        euler=[0, 0, 0.0],
        size=[COURSE_LENGTH, FLOOR_WIDTH, 1],
    )

    MIN_GAP = 1
    MAX_GAP = 3
    last_x = FLOOR_WIDTH / 2.0

    OBSTACLE_HEIGHT = 5.0
    OBSTACLE_WIDTH = 0.5

    for i in range(OBSTACLES):
        gap = np.random.uniform(MIN_GAP, MAX_GAP)
        is_left = np.random.choice([True, False])
        new_x = last_x + gap
        new_y = FLOOR_WIDTH / 4.0 if is_left else -FLOOR_WIDTH / 4.0
        tg.AddBox(
            position=[new_x, new_y, BASE_HEIGHT],
            euler=[0, 0, 0],
            size=[OBSTACLE_WIDTH, FLOOR_WIDTH / 2.0, OBSTACLE_HEIGHT],
        )

        last_x = new_x + gap

    tg.Save()


def gap_course(tg: TerrainGenerator) -> None:
    # starting floor
    BASE_HEIGHT = -0.5
    FLOOR_WIDTH = 5.0
    tg.AddBox(
        position=[0.0, 0.0, BASE_HEIGHT], euler=[0, 0, 0.0], size=[5, FLOOR_WIDTH, 1]
    )

    # randomized gaps
    HOLES = 100
    MIN_GAP = 0.3
    MAX_GAP = 1.0
    MIN_FLOOR_LEN = 2.5
    MAX_FLOOR_LEN = 6.0
    last_x = FLOOR_WIDTH / 2.0
    for i in range(HOLES):
        floor_len = np.random.uniform(MIN_FLOOR_LEN, MAX_FLOOR_LEN)
        gap_size = np.random.uniform(MIN_GAP, MAX_GAP)
        new_center_x = last_x + gap_size + (floor_len / 2.0)
        tg.AddBox(
            position=[new_center_x, 0.0, BASE_HEIGHT],
            euler=[0, 0, 0.0],
            size=[floor_len, FLOOR_WIDTH, 1],
        )
        last_x = new_center_x + (floor_len / 2.0)

    tg.Save()


if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    parent_path = os.path.dirname(current_path)
    models_path = os.path.join(parent_path, "models")
    tg = TerrainGenerator(
        f"{models_path}/anybotics_anymal_b", "scene_base.xml", "scene_corridor.xml"
    )
    obstacle_corridor(tg)
