from app import create_app


EVENT_ID = (
    "suspected_no_vest_T176_F149"
)


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

    cases = [
        "/api/health",
        "/api/events",
        "/api/stats",
        "/api/tracks/176",
        f"/api/events/{EVENT_ID}",
    ]

    failed = False

    for url in cases:
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

    for phase in (
        "pre",
        "open",
        "post",
    ):
        url = (
            f"/api/events/{EVENT_ID}"
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
        f"/api/events/{EVENT_ID}/clip"
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
