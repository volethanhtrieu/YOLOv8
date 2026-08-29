from app import create_app


def has_legacy_glass_field(value):
    if isinstance(value, dict):
        return any(
            "glass" in str(key).lower()
            or has_legacy_glass_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_legacy_glass_field(item) for item in value)
    return False


def main():
    app = create_app()
    app.config.update(
        TESTING=True
    )

    client = app.test_client()

    base_cases = [
        "/api/health",
        "/api/stats",
    ]

    failed = False

    for url in base_cases:
        response = client.get(url)

        print(
            response.status_code,
            url,
        )

        if response.status_code != 200:
            failed = True
        elif response.is_json and has_legacy_glass_field(response.get_json()):
            print("FAIL: legacy glass field exposed")
            failed = True

    event_response = client.get(
        "/api/events"
    )
    print(
        event_response.status_code,
        "/api/events",
    )

    event_payload = (
        event_response.get_json()
        if event_response.is_json
        else {}
    )
    events = event_payload.get(
        "items",
        [],
    )

    if event_response.status_code != 200:
        failed = True
    elif has_legacy_glass_field(event_payload):
        print("FAIL: legacy glass field exposed")
        failed = True

    if not events:
        print(
            "SKIP event detail/evidence/clip: "
            "published event store is empty."
        )
    else:
        event = events[0]
        event_id = str(
            event.get(
                "event_id",
                "",
            )
        )
        track_id = event.get(
            "track_id"
        )

        detail_cases = [
            f"/api/events/{event_id}",
        ]
        if track_id is not None:
            detail_cases.append(
                f"/api/tracks/{track_id}"
            )

        for url in detail_cases:
            response = client.get(url)
            print(response.status_code, url)
            if response.status_code != 200:
                failed = True
            elif (
                response.is_json
                and has_legacy_glass_field(
                    response.get_json()
                )
            ):
                print("FAIL: legacy glass field exposed")
                failed = True

        for phase in (
            "pre",
            "open",
            "post",
        ):
            url = (
                f"/api/events/{event_id}"
                f"/evidence"
                f"?phase={phase}"
                f"&view=crop"
            )
            response = client.get(url)
            print(
                response.status_code,
                url,
                response.content_type,
                "bytes=",
                len(response.data),
            )
            if (
                response.status_code != 200
                or response.content_type
                != "image/jpeg"
            ):
                failed = True

        clip_url = (
            f"/api/events/{event_id}/clip"
        )
        clip_response = client.get(
            clip_url
        )
        print(
            clip_response.status_code,
            clip_url,
            clip_response.content_type,
            "bytes=",
            len(clip_response.data),
        )
        if (
            clip_response.status_code != 200
            or clip_response.content_type
            != "video/mp4"
        ):
            failed = True

    if failed:
        raise SystemExit(
            "API V3 smoke test failed."
        )

    print()
    print(
        "ALL API V3 SMOKE TESTS PASSED"
    )


if __name__ == "__main__":
    main()
