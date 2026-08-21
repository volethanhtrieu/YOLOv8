from app import create_app


EVENT_ID = (
    "suspected_no_vest_T176_F149"
)


def main():
    app = create_app()
    app.config.update(
        TESTING=True
    )

    client = app.test_client()

    failed = False

    response = client.get(
        "/api/events"
    )

    print(
        response.status_code,
        "GET /api/events",
    )

    payload = response.get_json()
    items = payload.get(
        "items",
        [],
    )

    event = next(
        (
            item
            for item in items
            if item.get(
                "event_id"
            )
            == EVENT_ID
        ),
        None,
    )

    if event is None:
        print(
            "Expected restored event "
            "was not found."
        )
        failed = True
    else:
        print(
            "AI status:",
            event.get(
                "status"
            ),
        )
        print(
            "Human decision:",
            event.get(
                "human_decision"
            ),
        )
        print(
            "Final disposition:",
            event.get(
                "final_disposition"
            ),
        )

        if (
            event.get(
                "final_disposition"
            )
            != "FALSE_ALARM"
        ):
            failed = True

    response = client.get(
        "/api/review-queue"
        "?scope=published"
    )

    print(
        response.status_code,
        "GET /api/review-queue",
    )

    queue = response.get_json()

    print(
        "Queue total:",
        queue.get(
            "total"
        ),
    )

    if (
        response.status_code
        != 200
        or queue.get(
            "total"
        )
        != 0
    ):
        failed = True

    response = client.get(
        "/api/stats"
    )

    print(
        response.status_code,
        "GET /api/stats",
    )

    review_stats = (
        response.get_json()
        .get(
            "human_review",
            {},
        )
    )

    print(
        "Final stats:",
        review_stats.get(
            "by_final_disposition"
        ),
    )

    if failed:
        raise SystemExit(
            "Human Review V3 smoke "
            "test failed."
        )

    print()
    print(
        "ALL HUMAN REVIEW V3 "
        "SMOKE TESTS PASSED"
    )


if __name__ == "__main__":
    main()
