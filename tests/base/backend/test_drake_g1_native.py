from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest


def _native_extension_built() -> bool:
    try:
        return importlib.util.find_spec("drake_uni.compiled._drake_env_pool") is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _native_extension_built(), reason="native DrakeUni extension required")
def test_g1_drake_actuator_columns_match_source_model() -> None:
    """The real adapter preserves named control columns across sim2sim backends."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json
                import mujoco
                import numpy as np
                from unilab.assets import ASSETS_ROOT_PATH
                from unilab.base.backend_factory import create_backend
                from unilab.base.scene import SceneCfg

                scene = SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots/g1/scene_flat.xml"))
                # MuJoCo only parses the held-out source model here; no training samples.
                source = mujoco.MjModel.from_xml_path(scene.model_file)
                backend = create_backend(
                    "drake", scene, 2, 0.002,
                    drake_backend_mode="batch", drake_nthread=1,
                )
                expected_names = tuple(source.actuator(i).name for i in range(source.nu))
                expected_joints = tuple(
                    source.joint(int(source.actuator_trnid[i, 0])).name
                    for i in range(source.nu)
                )
                assert source.nu == backend.num_actuators == 29
                assert backend.get_actuator_names() == expected_names
                assert backend.get_actuator_joint_names() == expected_joints
                assert len(set(expected_names)) == 29
                assert all(expected_names) and all(expected_joints)
                np.testing.assert_allclose(
                    backend.get_keyframe_qpos("stand"), source.keyframe("stand").qpos,
                )
                print(json.dumps({"actuators": 29, "source_order_matches": True}))
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary == {"actuators": 29, "source_order_matches": True}
