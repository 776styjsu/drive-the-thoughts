"""Optional browser regression: uv run --with playwright pytest this_file -q."""

import argparse
import base64
import functools
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_frame_selector import build, cot_id


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def check_embedded_tour(page):
    # Keep both a committed selection and an unfinished edit through every step.
    page.locator("#reason").fill("An unfinished inclusion reason")
    before = page.evaluate("JSON.stringify(localStorage)")
    frame = page.locator("#frame-id").inner_text()
    summary = page.locator("#summary").inner_text()
    expected = [
        "#dashboard > div:first-child",
        "#category-focus",
        "#scene-controls",
        "#frame-image",
        "#decision-navigation",
        "#cot",
        "#category-editor",
        "#selection-reason",
        "#selection-actions",
        "#selected-panel",
        "#session-backup",
        "#export",
    ]
    for viewport in [{"width": 1280, "height": 900}, {"width": 390, "height": 844}]:
        page.set_viewport_size(viewport)
        assert (
            page.evaluate("document.documentElement.scrollWidth") <= viewport["width"]
        )
        page.locator("#start-tour").click()
        assert page.locator("#selector-tour").is_visible()
        assert page.locator("#tour-back").is_disabled()
        page.locator("#tour-next").click()
        page.locator("#tour-back").click()
        for index, target in enumerate(expected):
            assert page.locator("#selector-tour").get_attribute("data-target") == target
            assert page.locator("#tour-count").text_content() == (
                f"Step {index + 1} of {len(expected)}"
            )
            card = page.locator("#tour-card").bounding_box()
            outline = page.locator("#tour-highlight").bounding_box()
            area = page.locator(target).bounding_box()
            assert card and outline and area
            assert 0 <= card["x"] <= viewport["width"] - card["width"] + 1
            assert 0 <= card["y"] <= viewport["height"] - card["height"] + 1
            assert outline["x"] < area["x"] + area["width"]
            assert outline["x"] + outline["width"] > area["x"]
            assert outline["y"] < area["y"] + area["height"]
            assert outline["y"] + outline["height"] > area["y"]
            assert (
                outline["x"] + outline["width"] <= card["x"]
                or card["x"] + card["width"] <= outline["x"]
                or outline["y"] + outline["height"] <= card["y"]
                or card["y"] + card["height"] <= outline["y"]
            ), f"Tour card obscures the highlight for {target}"
            page.locator("#tour-next").click()
        assert not page.locator("#selector-tour").is_visible()
        assert page.evaluate("document.activeElement.id") == "start-tour"
        assert page.evaluate("JSON.stringify(localStorage)") == before
        assert page.locator("#frame-id").inner_text() == frame
        assert page.locator("#summary").inner_text() == summary
        assert page.locator("#reason").input_value() == "An unfinished inclusion reason"
    page.set_viewport_size({"width": 1280, "height": 900})
    for method in ["escape", "button"]:
        page.locator("#start-tour").click()
        for _ in range(5):
            page.keyboard.press("Tab")
            assert page.evaluate(
                "document.querySelector('#selector-tour').contains(document.activeElement)"
            )
        if method == "escape":
            page.keyboard.press("Escape")
        else:
            page.locator("#tour-close").click()
        assert not page.locator("#selector-tour").is_visible()
        assert page.evaluate("JSON.stringify(localStorage)") == before


def test_selection_replacement_session_and_annotation_export(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir()
    sid = "clipgt-00000000-0000-0000-0000-000000000001"
    missing_sid = "clipgt-00000000-0000-0000-0000-000000000002"
    (root / f"scene_2_{missing_sid}").mkdir()
    categories = {
        "longitudinal": "speed_adaptation_road_events",
        "lateral": "lane_keeping_centering",
        "causal_factors": ["road_events"],
    }
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1kAAAAASUVORK5CYII="
    )
    for index in [1, 2]:
        path = root / f"scene_1_{sid}" / "extracted_frames" / f"{sid}_t{index:04}"
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "clip_id": path.name,
                    "chain_of_thought": f"Slow for bend {index}",
                    "time_now_us": 1000000 + index * 100000,
                    "trajectory_poses": [{"x": 0}],
                }
            )
        )
        (path / "camera_front_wide_120fov.jpg").write_bytes(png)
    baseline = tmp_path / "benchmark.json"
    baseline.write_text("[]")
    output = tmp_path / "page"
    build(
        argparse.Namespace(
            rollouts=root,
            output_dir=output,
            benchmark=baseline,
            exclude_scenes=[],
            cot_labels=None,
            target_n=20,
            clip_step_s=0.1,
            video_offset_s=0,
            title="Selection test",
        )
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=output)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            browser = p.firefox.launch(
                headless=True,
                executable_path=os.environ.get("PLAYWRIGHT_FIREFOX_EXECUTABLE"),
            )
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.locator("#selector-name").fill("test annotator")
            page.locator("#selector-name").blur()
            page.locator("#choose").click()
            assert "confirmation" in page.locator("#status").inner_text()
            suggestion = tmp_path / "suggestions.json"
            suggestion.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "labels": [
                            {
                                "cot_id": cot_id("Slow for bend 1"),
                                "categories": categories,
                                "source": "test model",
                            }
                        ],
                    }
                )
            )
            page.locator("#labels-file").set_input_files(suggestion)
            page.wait_for_function(
                "document.querySelector('#longitudinal').value==='speed_adaptation_road_events'"
            )
            page.locator("#confirm-tags").check()
            page.locator("#reason").fill("Rare speed adaptation")
            page.locator("#choose").click()
            assert "1 chosen clips" in page.locator("#summary").inner_text()
            row = page.locator(
                "tr",
                has=page.locator(
                    '[data-target="longitudinal:speed_adaptation_road_events"]'
                ),
            )
            assert row.locator("td").nth(2).inner_text() == "1"
            page.locator("#next").click()
            page.locator("#longitudinal").select_option("yield_agent_right_of_way")
            page.locator("#lateral").select_option("turn")
            page.locator('[data-causal="critical_objects"]').check()
            page.locator("#confirm-tags").check()
            page.locator("#reason").fill("Prefer a yielding case")
            page.locator("#choose").click()
            assert "1 chosen clips" in page.locator("#summary").inner_text()
            assert row.locator("td").nth(2).inner_text() == "0"
            with page.expect_download() as download:
                page.locator("#save").click()
            session = tmp_path / "session.json"
            download.value.save_as(session)
            saved = json.loads(session.read_text())
            assert len(saved["selected"]) == 1
            assert saved["events"][-1]["action"] == "replace_selection"
            page.locator("#remove").click()
            assert "0 chosen clips" in page.locator("#summary").inner_text()
            page.locator("#import-file").set_input_files(session)
            page.wait_for_function(
                "document.querySelector('#summary').textContent.includes('1 chosen clips')"
            )
            bad = {**saved, "dataset_id": "wrong"}
            badfile = tmp_path / "bad.json"
            badfile.write_text(json.dumps(bad))
            page.locator("#import-file").set_input_files(badfile)
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Import rejected')"
            )
            assert "1 chosen clips" in page.locator("#summary").inner_text()
            check_embedded_tour(page)
            page.reload()
            assert "1 chosen clips" in page.locator("#summary").inner_text()
            with page.expect_download() as download:
                page.locator("#export").click()
            export = tmp_path / "selected.json"
            download.value.save_as(export)
            data = json.loads(export.read_text())
            assert len(data["entries"]) == 1
            entry = data["entries"][0]
            assert entry["clip_id"] == sid + "_t0002"
            assert (
                entry["labels"]["cot_decision_label"]["high_level_decision"][
                    "longitudinal"
                ]
                == "yield_agent_right_of_way"
            )
            assert entry["selection"]["reason"] == "Prefer a yielding case"
            assert "planned_trajectory_safe" not in entry
            assert Path(entry["source_scene_file"]).is_file()
            page.locator(f'[data-scene="{missing_sid}"]').click()
            page.locator("#start-tour").click()
            visited = []
            while page.locator("#selector-tour").is_visible():
                visited.append(
                    page.locator("#selector-tour").get_attribute("data-target")
                )
                page.locator("#tour-next").click()
            assert "#scene-status" in visited
            assert "#category-editor" not in visited
            assert "#selection-actions" not in visited
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_guided_walkthrough_counts_replacement_and_demo_export(tmp_path):
    tutorial = (
        Path(__file__).resolve().parents[1] / "tools/frame_selector_tutorial.html"
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=tutorial.parent)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as p:
            browser = p.firefox.launch(
                headless=True,
                executable_path=os.environ.get("PLAYWRIGHT_FIREFOX_EXECUTABLE"),
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/{tutorial.name}")
            assert page.locator("#next").is_disabled()
            page.locator('[data-focus="lead_obstacle_following"]').click()
            assert page.locator("#next").is_disabled()
            page.locator('[data-focus="speed_adaptation_road_events"]').click()
            page.locator("#next").click()
            page.locator("#clip-c").click()
            assert "Extract" in page.locator("#feedback").inner_text()
            assert page.locator("#next").is_disabled()
            page.locator("#clip-a").click()
            page.locator("#next").click()
            page.locator("#video-position").fill("6.2")
            page.locator("#snap").click()
            assert "6.0 s" in page.locator("#frame-name").inner_text()
            page.locator("#next").click()
            page.locator("#confirm").check()
            assert page.locator("#next").is_disabled()
            page.locator("#longitudinal").select_option("speed_adaptation_road_events")
            page.locator("#confirm").check()
            page.locator("#next").click()
            page.locator("#choose").click()
            assert page.locator("#next").is_disabled()
            page.locator("#reason").fill("Add an independent road-bend case.")
            page.locator("#choose").click()
            speed = page.locator('[data-category="speed_adaptation_road_events"]')
            yielding = page.locator('[data-category="yield_agent_right_of_way"]')
            assert speed.locator(".added-count").inner_text() == "1"
            assert speed.locator(".gap").inner_text() == "1"
            assert page.locator("#selected-count").inner_text() == "1 clip selected"
            page.locator("#next").click()
            page.locator("#replace").click()
            assert speed.locator(".added-count").inner_text() == "0"
            assert yielding.locator(".added-count").inner_text() == "1"
            assert page.locator("#selected-count").inner_text() == "1 clip selected"
            assert page.locator("#pedestrian").is_visible()
            page.locator("#next").click()
            with page.expect_download() as download:
                page.locator("#export-example").click()
            example = tmp_path / "demo.json"
            download.value.save_as(example)
            payload = json.loads(example.read_text())
            assert payload["demo"] is True
            assert len(payload["entries"]) == 1
            assert payload["entries"][0]["clip_id"].endswith("_t0100")
            assert "planned_trajectory_safe" not in payload["entries"][0]
            assert payload["selection_history"][-1]["action"] == "replace"
            page.locator("#next").click()
            assert page.locator("#handoff-card").is_visible()
            assert page.locator("#next").is_hidden()
            assert page.evaluate("localStorage.length") == 0
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator('[data-step="0"]').click()
            assert "Start with" in page.locator("#lesson-title").inner_text()
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.locator("#restart").click()
            assert page.locator("#selected-count").inner_text() == "0 clips selected"
            assert page.locator("#next").is_disabled()
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
