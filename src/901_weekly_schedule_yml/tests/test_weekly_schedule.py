"""Tests for weekly_schedule_yml."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from icalendar import Calendar
import yaml

from weekly_schedule_yml.main import (
    DAY_MAP,
    build_pomodoro_cycle,
    generate_ical,
    main,
    parse_duration,
    parse_schedule,
    parse_time,
    split_pomodoro,
)


class TestWeeklyScheduleYml(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(parse_time("06:30"), datetime.time(6, 30, 0))
        self.assertEqual(parse_time("14:15:30"), datetime.time(14, 15, 30))
        self.assertEqual(parse_time(datetime.time(8, 0)), datetime.time(8, 0))
        with self.assertRaises(ValueError):
            parse_time(12345)

    def test_parse_duration(self):
        self.assertEqual(parse_duration(25), datetime.timedelta(minutes=25))
        self.assertEqual(parse_duration("50m"), datetime.timedelta(minutes=50))
        self.assertEqual(parse_duration("1h"), datetime.timedelta(hours=1))
        self.assertEqual(parse_duration("30s"), datetime.timedelta(seconds=30))
        self.assertEqual(parse_duration("500ms"), datetime.timedelta(milliseconds=500))
        self.assertEqual(parse_duration(datetime.timedelta(minutes=10)), datetime.timedelta(minutes=10))
        with self.assertRaises(ValueError):
            parse_duration("invalid_duration")

    def test_build_pomodoro_cycle(self):
        items = build_pomodoro_cycle(
            focus_sec=3000,
            break_sec=600,
            long_break_sec=1200,
            sessions=3,
            body="bib",
        )
        self.assertEqual(len(items), 6)
        self.assertEqual(items[0], {"type": "focus", "dur": 3000, "body": "bib"})
        self.assertEqual(items[1], {"type": "break", "dur": 600, "body": None})
        self.assertEqual(items[2], {"type": "focus", "dur": 3000, "body": "bib"})
        self.assertEqual(items[3], {"type": "break", "dur": 600, "body": None})
        self.assertEqual(items[4], {"type": "focus", "dur": 3000, "body": "bib"})
        self.assertEqual(items[5], {"type": "long_break", "dur": 1200, "body": None})

    def test_split_pomodoro(self):
        slot = {
            "from_sec": 8 * 3600,
            "to_sec": 11 * 3600,
            "body": "bib",
        }
        res = split_pomodoro(
            slot,
            focus=datetime.timedelta(minutes=50),
            break_dur=datetime.timedelta(minutes=10),
            long_break_dur=datetime.timedelta(minutes=20),
            sessions=3,
        )
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["from_sec"], 8 * 3600)
        self.assertEqual(res[0]["to_sec"], 8 * 3600 + 50 * 60)
        self.assertEqual(res[1]["from_sec"], 9 * 3600)
        self.assertEqual(res[1]["to_sec"], 9 * 3600 + 50 * 60)
        self.assertEqual(res[2]["from_sec"], 10 * 3600)
        self.assertEqual(res[2]["to_sec"], 10 * 3600 + 50 * 60)

    def test_split_pomodoro_with_break_body(self):
        slot = {
            "from_sec": 8 * 3600,
            "to_sec": 10 * 3600,
            "body": "focus",
        }
        res = split_pomodoro(
            slot,
            focus=datetime.timedelta(minutes=50),
            break_dur=datetime.timedelta(minutes=10),
            long_break_dur=datetime.timedelta(minutes=20),
            sessions=2,
            break_body="short break",
            long_break_body="long break",
        )
        # Cycle: focus(50m), break(10m), focus(50m), long_break(20m clamped to 10m)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0]["body"], "focus")
        self.assertEqual(res[1]["body"], "short break")
        self.assertEqual(res[2]["body"], "focus")
        self.assertEqual(res[3]["body"], "long break")
        self.assertEqual(res[3]["from_sec"], 8 * 3600 + 110 * 60)
        self.assertEqual(res[3]["to_sec"], 10 * 3600)

    def test_split_pomodoro_with_offset(self):
        slot = {
            "from_sec": 8 * 3600,
            "to_sec": 9 * 3600,
            "body": "bib",
        }
        # offset 25m into a 50m focus session: first slot should run for 25m (from 08:00 to 08:25)
        res = split_pomodoro(
            slot,
            focus=datetime.timedelta(minutes=50),
            break_dur=datetime.timedelta(minutes=10),
            long_break_dur=datetime.timedelta(minutes=20),
            sessions=3,
            offset=datetime.timedelta(minutes=25),
        )
        self.assertEqual(res[0]["from_sec"], 8 * 3600)
        self.assertEqual(res[0]["to_sec"], 8 * 3600 + 25 * 60)

    def test_parse_schedule_against_yaml(self):
        yaml_path = Path(__file__).resolve().parents[1] / "weekly_schedule.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        days_slots, meta = parse_schedule(raw_data)
        self.assertIn("mon", days_slots)
        self.assertEqual(len(days_slots["mon"]), 13)
        self.assertEqual(len(days_slots["tue"]), 10)
        self.assertEqual(len(days_slots["wed"]), 11)
        self.assertEqual(len(days_slots["thu"]), 12)
        self.assertEqual(len(days_slots["fri"]), 10)
        self.assertEqual(len(days_slots["sat"]), 6)
        self.assertEqual(len(days_slots["sun"]), 5)
        total_slots = sum(len(slots) for slots in days_slots.values())
        self.assertEqual(total_slots, 67)

    def test_match_typst_evaluation_exactly(self):
        """Verify that Python output produces the exact same slots as weekly_schedule.typ."""
        typst_bin = shutil.which("typst")
        typ_path = Path("/home/noah/Documents/repos/notes/src/00_organization/00_time_plan/config/weekly_schedule.typ")
        if not typst_bin or not typ_path.exists():
            self.skipTest("typst binary or weekly_schedule.typ not found")

        yaml_path = Path(__file__).resolve().parents[1] / "weekly_schedule.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        py_days, _ = parse_schedule(raw_data)

        cmd = [
            "typst", "eval", "--root", "/home/noah/Documents/repos/notes",
            'import "/src/00_organization/00_time_plan/config/weekly_schedule.typ": days; repr(days)'
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        typst_repr = json.loads(res.stdout)

        day_re = re.compile(r"(\w+):\s*\((.*?)\n  \)", re.DOTALL)
        slot_re = re.compile(
            r"from:\s*datetime\(hour:\s*(\d+),\s*minute:\s*(\d+),\s*second:\s*(\d+)\),\s*"
            r"to:\s*datetime\(hour:\s*(\d+),\s*minute:\s*(\d+),\s*second:\s*(\d+)\),\s*"
            r"body:\s*\[(.*?)\]",
            re.DOTALL
        )

        typst_days: dict[str, list[dict[str, Any]]] = {}
        for day_m in day_re.finditer(typst_repr):
            day_name = day_m.group(1)
            slots_text = day_m.group(2)
            slots = []
            for s_m in slot_re.finditer(slots_text):
                f_h, f_m, f_s, t_h, t_m, t_s, b = s_m.groups()
                slots.append({
                    "from_sec": int(f_h) * 3600 + int(f_m) * 60 + int(f_s),
                    "to_sec": int(t_h) * 3600 + int(t_m) * 60 + int(t_s),
                    "body": b.strip()
                })
            typst_days[day_name] = slots

        for day in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]:
            self.assertEqual(
                len(py_days[day]),
                len(typst_days[day]),
                f"Slot count mismatch for {day}",
            )
            for idx, (py_slot, typ_slot) in enumerate(zip(py_days[day], typst_days[day])):
                self.assertEqual(
                    py_slot["from_sec"],
                    typ_slot["from_sec"],
                    f"from_sec mismatch on {day} slot {idx}",
                )
                self.assertEqual(
                    py_slot["to_sec"],
                    typ_slot["to_sec"],
                    f"to_sec mismatch on {day} slot {idx}",
                )
                self.assertEqual(
                    py_slot["body"],
                    typ_slot["body"],
                    f"body mismatch on {day} slot {idx}",
                )

    def test_generate_ical(self):
        days_slots = {
            "mon": [
                {"from_sec": 8 * 3600, "to_sec": 9 * 3600, "body": "Study"}
            ]
        }
        meta = {"start_date": "2026-09-14"}
        cal = generate_ical(days_slots, meta)
        parsed_cal = Calendar.from_ical(cal.to_ical())
        events = [c for c in parsed_cal.subcomponents if c.name == "VEVENT"]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(str(event["summary"]), "Study")
        self.assertEqual(event["rrule"]["FREQ"], ["WEEKLY"])
        self.assertEqual(event["rrule"]["BYDAY"], ["MO"])
        self.assertEqual(event["dtstart"].dt, datetime.datetime(2026, 9, 14, 8, 0))
        self.assertEqual(event["dtend"].dt, datetime.datetime(2026, 9, 14, 9, 0))

    def test_deterministic_uids(self):
        days_slots = {
            "mon": [
                {"from_sec": 8 * 3600, "to_sec": 9 * 3600, "body": "Study"}
            ]
        }
        meta = {"start_date": "2026-09-14"}
        cal1 = generate_ical(days_slots, meta)
        cal2 = generate_ical(days_slots, meta)
        ev1 = [c for c in cal1.subcomponents if c.name == "VEVENT"][0]
        ev2 = [c for c in cal2.subcomponents if c.name == "VEVENT"][0]
        self.assertEqual(str(ev1["uid"]), str(ev2["uid"]))

    def test_cli_execution_with_output_flag(self):
        yaml_path = Path(__file__).resolve().parents[1] / "weekly_schedule.yaml"
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_out.ics"
            main([str(yaml_path), "-o", str(out_file)])
            self.assertTrue(out_file.exists())
            cal = Calendar.from_ical(out_file.read_bytes())
            events = [c for c in cal.subcomponents if c.name == "VEVENT"]
            self.assertEqual(len(events), 67)

    def test_cli_execution_default_output(self):
        yaml_path = Path(__file__).resolve().parents[1] / "weekly_schedule.yaml"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_yaml = Path(tmp_dir) / "schedule.yaml"
            shutil.copy(yaml_path, tmp_yaml)
            main([str(tmp_yaml)])
            expected_ics = Path(tmp_dir) / "schedule.ics"
            self.assertTrue(expected_ics.exists())
            cal = Calendar.from_ical(expected_ics.read_bytes())
            events = [c for c in cal.subcomponents if c.name == "VEVENT"]
            self.assertEqual(len(events), 67)

    def test_cli_file_not_found(self):
        with self.assertRaises(SystemExit):
            main(["non_existent_file.yaml"])


if __name__ == "__main__":
    unittest.main()
