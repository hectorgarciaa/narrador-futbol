import math
import unittest

from football_ai.positions.pipeline.online import OnlineSpecialSeedRoleAssigner


class SegmentAssignmentRegressionTests(unittest.TestCase):
    def test_segment_hungarian_keeps_tracks_with_same_segment_id_separate(self):
        assigner = OnlineSpecialSeedRoleAssigner.__new__(OnlineSpecialSeedRoleAssigner)
        assigner.segment_expected_slots_by_team = {"Team A": ["LI", "MI"]}
        assigner.expected_roles_by_team = {"Team A": ["LI", "MI"]}
        assigner.stats = {"segment_level_assignments": 0}
        assigner.segment_assignment_rows = []

        state_a = {
            "track_id": 3,
            "segment_id": 1,
            "majority_role": "LD",
            "majority_expected_role_slot": "LD",
            "observations": 12,
            "player_name": None,
            "lineup_slot": None,
        }
        state_b = {
            "track_id": 4,
            "segment_id": 1,
            "majority_role": "MC",
            "majority_expected_role_slot": "MC",
            "observations": 11,
            "player_name": None,
            "lineup_slot": None,
        }
        track_a = {}
        track_b = {}

        def fake_assign_segments_for_team(team_id, segment_states, expected_slots):
            self.assertEqual(team_id, "Team A")
            self.assertEqual(segment_states, [state_a, state_b])
            self.assertEqual(expected_slots, ["LI", "MI"])
            return [
                {"state": state_a, "slot": "LI", "cost": 0.1},
                {"state": state_b, "slot": "MI", "cost": 0.2},
            ]

        def fake_apply_segment_assignment_to_track(track_data, state, slot_name, frame_id, assignment_cost):
            track_data["display_role_slot"] = slot_name
            track_data["frame_id"] = frame_id
            track_data["assignment_cost"] = assignment_cost

        assigner._assign_segments_for_team = fake_assign_segments_for_team
        assigner._apply_segment_assignment_to_track = fake_apply_segment_assignment_to_track

        assigner._apply_active_segment_assignments(
            {"Team A": [(state_a, track_a), (state_b, track_b)]},
            frame_id=123,
        )

        self.assertEqual(track_a["display_role_slot"], "LI")
        self.assertEqual(track_b["display_role_slot"], "MI")
        self.assertTrue(math.isclose(track_a["assignment_cost"], 0.1))
        self.assertTrue(math.isclose(track_b["assignment_cost"], 0.2))
        self.assertEqual(assigner.stats["segment_level_assignments"], 2)
        self.assertEqual(
            [row["display_role_slot"] for row in assigner.segment_assignment_rows],
            ["LI", "MI"],
        )


if __name__ == "__main__":
    unittest.main()
