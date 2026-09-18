"""Convert a weekly schedule YAML file into an iCalendar (.ics) file with weekly recurring slots."""

from __future__ import annotations

import argparse
import datetime
import sys
import uuid
import zoneinfo
from pathlib import Path
from typing import Any

from icalendar import Calendar, Event
import yaml

DAY_MAP: dict[str, tuple[int, str]] = {
    "mon": (0, "MO"),
    "monday": (0, "MO"),
    "tue": (1, "TU"),
    "tuesday": (1, "TU"),
    "wed": (2, "WE"),
    "wednesday": (2, "WE"),
    "thu": (3, "TH"),
    "thursday": (3, "TH"),
    "fri": (4, "FR"),
    "friday": (4, "FR"),
    "sat": (5, "SA"),
    "saturday": (5, "SA"),
    "sun": (6, "SU"),
    "sunday": (6, "SU"),
}


def parse_time(val: Any) -> datetime.time:
    """Parse time representations (e.g. '06:30', '14:15:00', datetime.time) into datetime.time."""
    if isinstance(val, datetime.time):
        return val
    if isinstance(val, str):
        parts = val.strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        s = int(parts[2]) if len(parts) > 2 else 0
        return datetime.time(h, m, s)
    raise ValueError(f"Unknown time format: {val!r}")


def parse_duration(d: Any) -> datetime.timedelta:
    """Parse duration representations (e.g. '50m', '1h', '20s', int minutes, timedelta)."""
    if isinstance(d, datetime.timedelta):
        return d
    if isinstance(d, (int, float)):
        return datetime.timedelta(minutes=float(d))
    if isinstance(d, str):
        s = d.strip()
        if s.endswith("ms"):
            return datetime.timedelta(milliseconds=float(s[:-2]))
        elif s.endswith("m"):
            return datetime.timedelta(minutes=float(s[:-1]))
        elif s.endswith("h"):
            return datetime.timedelta(hours=float(s[:-1]))
        elif s.endswith("s"):
            return datetime.timedelta(seconds=float(s[:-1]))
        else:
            return datetime.timedelta(minutes=float(s))
    raise ValueError(f"Unknown duration format: {d!r}")


def build_pomodoro_cycle(
    focus_sec: int,
    break_sec: int,
    long_break_sec: int,
    sessions: int,
    body: str,
    break_body: str | None = None,
    long_break_body: str | None = None,
) -> list[dict[str, Any]]:
    """Build a single Pomodoro cycle items list matching pomodoro.typ logic."""
    items: list[dict[str, Any]] = []
    pair_count = max(0, sessions - 1)

    # 1. Initial focus session
    items.append({"type": "focus", "dur": focus_sec, "body": body})

    # 2. (sessions - 1) pairs of (break, focus)
    for _ in range(pair_count):
        if break_sec > 0:
            items.append({"type": "break", "dur": break_sec, "body": break_body})
        items.append({"type": "focus", "dur": focus_sec, "body": body})

    # 3. Long break after sessions focus sessions
    if long_break_sec > 0:
        items.append({"type": "long_break", "dur": long_break_sec, "body": long_break_body})

    return items


def split_pomodoro(
    slot: dict[str, Any],
    focus: datetime.timedelta,
    break_dur: datetime.timedelta,
    long_break_dur: datetime.timedelta,
    sessions: int,
    offset: datetime.timedelta = datetime.timedelta(0),
    break_body: str | None = None,
    long_break_body: str | None = None,
) -> list[dict[str, Any]]:
    """Split a slot into Pomodoro intervals based on pomodoro.typ."""
    focus_sec = int(focus.total_seconds())
    break_sec = int(break_dur.total_seconds())
    long_break_sec = int(long_break_dur.total_seconds())
    slot_from = slot["from_sec"]
    slot_to = slot["to_sec"]

    if focus_sec <= 0 or slot_from >= slot_to:
        return []

    items = build_pomodoro_cycle(
        focus_sec=focus_sec,
        break_sec=break_sec,
        long_break_sec=long_break_sec,
        sessions=sessions,
        body=slot["body"],
        break_body=break_body,
        long_break_body=long_break_body,
    )

    cycle_sec = sum(item["dur"] for item in items)
    if cycle_sec <= 0:
        return []

    # Shift timeline according to offset
    off_sec = int(offset.total_seconds())
    shift_sec = ((off_sec % cycle_sec) + cycle_sec) % cycle_sec
    cur = slot_from - shift_sec

    result: list[dict[str, Any]] = []
    done = False

    while not done:
        for it in items:
            it_start = cur
            it_end = cur + it["dur"]
            cur = it_end

            # Skip sessions completely before slot start
            if it_end <= slot_from:
                continue
            # Stop once sessions start past slot end
            if it_start >= slot_to:
                done = True
                break

            # Clamp interval to [slot_from, slot_to]
            s = max(it_start, slot_from)
            e = min(it_end, slot_to)

            if e > s and it["body"] is not None:
                new_slot = dict(slot)
                new_slot["from_sec"] = s
                new_slot["to_sec"] = e
                new_slot["body"] = it["body"]
                result.append(new_slot)

            if it_end >= slot_to:
                done = True
                break

    return result


def parse_schedule(raw_data: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Parse raw YAML data into processed daily slots and schedule metadata."""
    pomo_cfg = raw_data.get("pomodoro") or {}

    default_rules = {
        "focus": parse_duration(pomo_cfg.get("focus", 25)),
        "break_dur": parse_duration(pomo_cfg.get("break", pomo_cfg.get("short_break", 5))),
        "long_break_dur": parse_duration(pomo_cfg.get("long_break", 15)),
        "sessions": int(pomo_cfg.get("sessions", 4)),
        "offset": parse_duration(pomo_cfg.get("offset", 0)),
        "break_body": pomo_cfg.get("break_body"),
        "long_break_body": pomo_cfg.get("long_break_body"),
    }

    schedule_data = raw_data.get("schedule", raw_data)
    if not isinstance(schedule_data, dict):
        schedule_data = raw_data

    days_slots: dict[str, list[dict[str, Any]]] = {}

    for day_key, entries in schedule_data.items():
        if not isinstance(day_key, str) or day_key.lower() not in DAY_MAP:
            continue
        canon_day = day_key.lower()
        if not isinstance(entries, list):
            continue

        day_list: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            from_t = parse_time(entry["from"])
            to_t = parse_time(entry["to"])
            from_sec = from_t.hour * 3600 + from_t.minute * 60 + from_t.second
            to_sec = to_t.hour * 3600 + to_t.minute * 60 + to_t.second

            body = str(entry.get("body") or entry.get("summary") or entry.get("title") or "")
            slot = {
                "from_sec": from_sec,
                "to_sec": to_sec,
                "body": body,
                "raw_entry": entry,
            }

            pomo = entry.get("pomodoro", False)
            if pomo:
                rules = dict(default_rules)
                extra_args: dict[str, Any] = {}
                if isinstance(pomo, dict):
                    if "focus" in pomo:
                        rules["focus"] = parse_duration(pomo["focus"])
                    if "break" in pomo:
                        rules["break_dur"] = parse_duration(pomo["break"])
                    elif "short_break" in pomo:
                        rules["break_dur"] = parse_duration(pomo["short_break"])
                    if "long_break" in pomo:
                        rules["long_break_dur"] = parse_duration(pomo["long_break"])
                    if "sessions" in pomo:
                        rules["sessions"] = int(pomo["sessions"])
                    if "offset" in pomo:
                        extra_args["offset"] = parse_duration(pomo["offset"])
                    if "break_body" in pomo:
                        rules["break_body"] = pomo["break_body"]
                    if "long_break_body" in pomo:
                        rules["long_break_body"] = pomo["long_break_body"]

                if "offset" in entry:
                    extra_args["offset"] = parse_duration(entry["offset"])

                day_list.extend(split_pomodoro(slot, **rules, **extra_args))
            else:
                day_list.append(slot)

        days_slots[canon_day] = day_list

    return days_slots, raw_data


def generate_ical(
    days_slots: dict[str, list[dict[str, Any]]],
    meta: dict[str, Any],
    cal_name: str = "Weekly Schedule",
) -> Calendar:
    """Generate an iCalendar object with weekly recurring events."""
    cal = Calendar()
    cal.add("prodid", "-//Weekly Schedule YML//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")

    name = meta.get("title") or meta.get("name") or cal_name
    cal.add("x-wr-calname", name)

    tz: datetime.tzinfo | None = None
    tz_str = meta.get("timezone") or meta.get("tz")
    if tz_str:
        tz = zoneinfo.ZoneInfo(str(tz_str))
        cal.add("x-wr-timezone", str(tz_str))

    # Determine reference Monday date
    start_date_val = meta.get("start_date") or meta.get("start") or meta.get("from")
    if start_date_val:
        if isinstance(start_date_val, datetime.datetime):
            ref_date = start_date_val.date()
        elif isinstance(start_date_val, datetime.date):
            ref_date = start_date_val
        elif isinstance(start_date_val, str):
            ref_date = datetime.date.fromisoformat(start_date_val)
        else:
            ref_date = datetime.date.today()
    else:
        ref_date = datetime.date.today()

    base_monday = ref_date - datetime.timedelta(days=ref_date.weekday())

    # Check for until / end date
    until_val = meta.get("until") or meta.get("end_date") or meta.get("end") or meta.get("to")
    until_dt: datetime.date | datetime.datetime | None = None
    if until_val:
        if isinstance(until_val, (datetime.datetime, datetime.date)):
            until_dt = until_val
        elif isinstance(until_val, str):
            until_dt = datetime.date.fromisoformat(until_val)

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    uid_namespace = uuid.UUID("f3c834aa-2521-41ee-a548-a006ee9808d4")

    for day_key, slots in days_slots.items():
        day_offset, byday_code = DAY_MAP[day_key]
        slot_date = base_monday + datetime.timedelta(days=day_offset)

        for slot in slots:
            from_sec = slot["from_sec"]
            to_sec = slot["to_sec"]

            start_dt = datetime.datetime.combine(slot_date, datetime.time.min) + datetime.timedelta(seconds=from_sec)
            end_dt = datetime.datetime.combine(slot_date, datetime.time.min) + datetime.timedelta(seconds=to_sec)
            if to_sec <= from_sec:
                end_dt += datetime.timedelta(days=1)

            if tz is not None:
                start_dt = start_dt.replace(tzinfo=tz)
                end_dt = end_dt.replace(tzinfo=tz)

            event = Event()
            summary = slot["body"]
            event.add("summary", summary)
            event.add("dtstart", start_dt)
            event.add("dtend", end_dt)

            rrule_dict: dict[str, Any] = {"freq": "weekly", "byday": byday_code}
            if until_dt is not None:
                rrule_dict["until"] = until_dt
            event.add("rrule", rrule_dict)

            # Deterministic UID for idempotency
            uid_key = f"{day_key}-{from_sec}-{to_sec}-{summary}"
            event_uid = f"{uuid.uuid5(uid_namespace, uid_key)}@weekly-schedule"
            event.add("uid", event_uid)
            event.add("dtstamp", now_utc)

            raw = slot.get("raw_entry", {})
            if "description" in raw:
                event.add("description", raw["description"])
            if "location" in raw:
                event.add("location", raw["location"])

            cal.add_component(event)

    return cal


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a weekly schedule YAML file into an iCalendar (.ics) file."
    )
    parser.add_argument(
        "yaml_file",
        type=Path,
        help="Path to the input YAML schedule file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to output .ics file (defaults to input file with .ics extension).",
    )

    args = parser.parse_args(argv)

    if not args.yaml_file.exists():
        parser.error(f"File not found: {args.yaml_file}")

    with open(args.yaml_file, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        parser.error(f"Invalid YAML structure in {args.yaml_file}: expected dictionary")

    days_slots, meta = parse_schedule(raw_data)
    cal_title = args.yaml_file.stem.replace("_", " ").title()
    cal = generate_ical(days_slots, meta, cal_name=cal_title)

    output_path = args.output or args.yaml_file.with_suffix(".ics")
    output_path.write_bytes(cal.to_ical())
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
