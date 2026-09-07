"""Controller state regressions without a headset or running Vuer server."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from televuer import TeleVuerWrapper


class ControllerStateTests(unittest.TestCase):
    def setUp(self):
        for target in ("televuer.televuer.Vuer", "televuer.televuer.Process"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_wrapper(self, use_hand_tracking):
        wrapper = TeleVuerWrapper(
            use_hand_tracking=use_hand_tracking,
            img_shape=(480, 1280),
            display_mode="pass-through",
        )
        self.addCleanup(wrapper.tvuer.close)
        return wrapper

    def assert_controller_state(self, tvuer, side, state):
        for name, default in (
            ("trigger", False), ("triggerValue", 0.0),
            ("squeeze", False), ("squeezeValue", 0.0),
            ("thumbstick", False), ("thumbstickValue", [0.0, 0.0]),
            ("aButton", False), ("bButton", False),
        ):
            np.testing.assert_equal(
                getattr(tvuer, f"{side}_ctrl_{name}"), state.get(name, default),
                err_msg=f"{side}_ctrl_{name}",
            )

    def test_controller_defaults_before_first_event(self):
        for hand_tracking in (True, False):
            with self.subTest(hand_tracking=hand_tracking):
                wrapper = self.make_wrapper(hand_tracking)
                data = wrapper.get_tele_data()
                for side in ("left", "right"):
                    self.assert_controller_state(wrapper.tvuer, side, {})
                    for button in ("aButton", "bButton"):
                        self.assertFalse(getattr(data, f"{side}_ctrl_{button}"))

    def test_controller_updates_and_releases_reach_wrapper(self):
        left = dict(trigger=True, triggerValue=0.25, squeeze=True,
                    squeezeValue=0.5, thumbstick=True, thumbstickValue=[0.25, -0.5],
                    aButton=True, bButton=False)
        right = dict(trigger=False, triggerValue=0.75, squeeze=False,
                     squeezeValue=1.0, thumbstick=False, thumbstickValue=[-0.75, 1.0],
                     aButton=False, bButton=True)
        pose = np.eye(4).flatten(order="F").tolist()
        for hand_tracking in (True, False):
            with self.subTest(hand_tracking=hand_tracking):
                wrapper = self.make_wrapper(hand_tracking)
                for left_state, right_state in ((left, right), (right, left), ({}, {})):
                    event = SimpleNamespace(value=dict(
                        left=pose, right=pose, leftState=left_state, rightState=right_state,
                    ))
                    asyncio.run(wrapper.tvuer.on_controller_move(event, None))
                    data = wrapper.get_tele_data()
                    for side, state in (("left", left_state), ("right", right_state)):
                        self.assert_controller_state(wrapper.tvuer, side, state)
                        for button in ("aButton", "bButton"):
                            self.assertEqual(
                                getattr(data, f"{side}_ctrl_{button}"), state.get(button, False),
                            )

    def test_controller_poses_only_update_in_controller_mode(self):
        left_hand, right_hand = np.eye(4), np.eye(4)
        left_hand[:3, 3] = [1.0, 2.0, 3.0]
        right_hand[:3, 3] = [-1.0, -2.0, -3.0]
        left_controller, right_controller = np.eye(4), np.eye(4)
        left_controller[:3, 3] = [4.0, 5.0, 6.0]
        right_controller[:3, 3] = [-4.0, -5.0, -6.0]
        for hand_tracking in (True, False):
            with self.subTest(hand_tracking=hand_tracking):
                tvuer = self.make_wrapper(hand_tracking).tvuer
                if hand_tracking:
                    event = SimpleNamespace(value=dict(
                        left=left_hand.flatten(order="F").tolist() * 25,
                        right=right_hand.flatten(order="F").tolist() * 25,
                        leftState={}, rightState={},
                    ))
                    asyncio.run(tvuer.on_hand_move(event, None))
                event = SimpleNamespace(value=dict(
                    left=left_controller.flatten(order="F").tolist(),
                    right=right_controller.flatten(order="F").tolist(),
                    leftState={"aButton": True}, rightState={"bButton": True},
                ))
                asyncio.run(tvuer.on_controller_move(event, None))
                np.testing.assert_array_equal(
                    tvuer.left_arm_pose, left_hand if hand_tracking else left_controller,
                )
                np.testing.assert_array_equal(
                    tvuer.right_arm_pose, right_hand if hand_tracking else right_controller,
                )


if __name__ == "__main__":
    unittest.main()
