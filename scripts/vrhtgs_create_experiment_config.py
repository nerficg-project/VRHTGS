#! /usr/bin/env python3

"""create_experiment_config.py: Creates shuffled experiment configs for the
perceptual study."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from random import sample, shuffle

import utils
with utils.DiscoverSourcePath():
    import Framework


@dataclass
class Condition:
    model_path: Path
    model_type: Literal['HTGS', 'VRHTGS']
    anti_aliasing: bool = False
    blur_periphery: bool = False
    periphery_interpolation_mode: Literal['nearest', 'linear'] = 'nearest'

    def __str__(self):
        return f'{self.model_type},{str(self.model_path.absolute())},{self.anti_aliasing},' \
               f'{self.blur_periphery},{self.periphery_interpolation_mode}'


def collect_conditions(
        parent_dir: Path, model_dir: str, *,
        model_type: Literal['HTGS', 'VRHTGS'],
        anti_aliasing: bool = False,
        blur_periphery: bool = True,
        periphery_interpolation_mode: Literal['linear', 'nearest'] = 'linear',
        excluded_conditions: list[Condition] | None = None,
) -> list[Condition]:
    conditions = []
    excluded_condition_names = {cond.model_path.name.split('_')[0] for cond in excluded_conditions} if excluded_conditions is not None else set()
    for model in (parent_dir / model_dir).iterdir():
        if not model.is_dir():
            continue
        if model.name.split('_')[0] in excluded_condition_names:
            continue

        condition = Condition(
            model_path=model,
            model_type=model_type,
            anti_aliasing=anti_aliasing,
            blur_periphery=blur_periphery,
            periphery_interpolation_mode=periphery_interpolation_mode,
        )
        conditions.append(condition)
    return conditions


def generate_calibration_mode_config(participant_id: int) -> None:
    conditions = []
    experiment_dir = Framework.Directories.OUTPUT_DIR / 'vr-scenes' / 'experiment'
    calibration_dir = Framework.Directories.OUTPUT_DIR / 'vr-scenes' / 'calibration'
    htgs_dir = 'htgs'

    conditions += collect_conditions(calibration_dir, htgs_dir, model_type='HTGS')
    conditions += collect_conditions(experiment_dir, htgs_dir, model_type='HTGS')

    # Write to file
    with open(Framework.Directories.OUTPUT_DIR / f'experiment_config_{participant_id}.csv', 'w') as f:
        for condition in conditions:
            f.write(f"{condition}\n")
    print(f'Wrote {len(conditions)} conditions to experiment_config_{participant_id}.csv for calibration mode')


def generate_experiment_config(participant_id: int) -> None:
    experiment_conditions = []
    calibration_conditions = []
    NUM_CALIBRATION_HTGS = 1
    NUM_CALIBRATION_FOVEATED = 1
    NUM_CALIBRATION_AA = 1
    NUM_CALIBRATION_BGM = 1

    experiment_dir = Framework.Directories.OUTPUT_DIR / 'vr-scenes' / 'experiment'
    calibration_dir = Framework.Directories.OUTPUT_DIR / 'vr-scenes' / 'calibration'
    htgs_dir = 'htgs'
    vr_htgs_dir = 'htgs'
    vr_htgs_aa_dir = 'htgs-fov'
    vr_htgs_aa_bgm_dir = 'htgs-bgm'

    BLUR_PERIPHERY = True
    PERIPHERY_INTERPOLATION_MODE: Literal['nearest', 'linear'] = 'linear'

    # Generate calibration conditions
    calibration_conditions += sample(collect_conditions(calibration_dir, htgs_dir, model_type='HTGS'),
                                     k=NUM_CALIBRATION_HTGS)
    calibration_conditions += sample(collect_conditions(calibration_dir, vr_htgs_dir, model_type='VRHTGS',
                                                        anti_aliasing=False, blur_periphery=BLUR_PERIPHERY, excluded_conditions=calibration_conditions,
                                                        periphery_interpolation_mode=PERIPHERY_INTERPOLATION_MODE), k=NUM_CALIBRATION_FOVEATED)
    calibration_conditions += sample(collect_conditions(calibration_dir, vr_htgs_aa_dir, model_type='VRHTGS',
                                                        anti_aliasing=True, blur_periphery=BLUR_PERIPHERY, excluded_conditions=calibration_conditions,
                                                        periphery_interpolation_mode=PERIPHERY_INTERPOLATION_MODE), k=NUM_CALIBRATION_AA)
    calibration_conditions += sample(collect_conditions(calibration_dir, vr_htgs_aa_bgm_dir, model_type='VRHTGS',
                                                        anti_aliasing=True, blur_periphery=BLUR_PERIPHERY, excluded_conditions=calibration_conditions,
                                                        periphery_interpolation_mode=PERIPHERY_INTERPOLATION_MODE), k=NUM_CALIBRATION_BGM)

    # Generate experiment conditions
    experiment_conditions += collect_conditions(experiment_dir, htgs_dir, model_type='HTGS')
    experiment_conditions += collect_conditions(experiment_dir, vr_htgs_dir, model_type='VRHTGS',
                                                anti_aliasing=False, blur_periphery=BLUR_PERIPHERY,
                                                periphery_interpolation_mode=PERIPHERY_INTERPOLATION_MODE)
    experiment_conditions += collect_conditions(experiment_dir, vr_htgs_aa_dir, model_type='VRHTGS',
                                                anti_aliasing=True, blur_periphery=BLUR_PERIPHERY,
                                                periphery_interpolation_mode=PERIPHERY_INTERPOLATION_MODE)
    experiment_conditions += collect_conditions(experiment_dir, vr_htgs_aa_bgm_dir, model_type='VRHTGS',
                                                anti_aliasing=True, blur_periphery=BLUR_PERIPHERY,
                                                periphery_interpolation_mode=PERIPHERY_INTERPOLATION_MODE)

    # Shuffle and write to file
    shuffle(experiment_conditions)
    shuffle(calibration_conditions)
    with open(Framework.Directories.OUTPUT_DIR / f'experiment_config_{participant_id}.csv', 'w') as f:
        for condition in calibration_conditions:
            f.write(f"{condition}\n")
        for condition in experiment_conditions:
            f.write(f"{condition}\n")
    print(f'Wrote {len(calibration_conditions) + len(experiment_conditions)} conditions to experiment_config_{participant_id}.csv')


def main(n: int = 10, calibration_config: bool = False) -> None:
    if calibration_config:
        generate_calibration_mode_config(-1)
    for i in range(n):
        generate_experiment_config(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates shuffled experiment configs for the perceptual study.')
    parser.add_argument('--num-experiments', type=int, default=10, help='Number of experiment configs to generate (default: 10)')
    parser.add_argument('--calibration-config', action='store_true', help='Also generate the calibration mode config (participant -1)')
    args = parser.parse_args()
    main(args.num_experiments, args.calibration_config)
