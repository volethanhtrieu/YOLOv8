from collections import Counter

from app import create_app


FINAL_BY_DECISION = {
    "CONFIRMED_VIOLATION": "CONFIRMED_VIOLATION",
    "FALSE_ALARM": "FALSE_ALARM",
    "NEEDS_REVIEW": "NEEDS_REVIEW",
}


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

    expected_final_counts = Counter()
    expected_queue_total = 0

    for event in items:
        human_decision = event.get(
            "human_decision"
        )
        expected_final = FINAL_BY_DECISION.get(
            human_decision,
            "PENDING_REVIEW",
        )
        actual_final = event.get(
            "final_disposition"
        )
        expected_final_counts[
            expected_final
        ] += 1

        review_state = str(
            event.get(
                "human_review_state",
                "UNREVIEWED",
            )
        )
        if review_state in {
            "UNREVIEWED",
            "NEEDS_REVIEW",
        }:
            expected_queue_total += 1

        print(
            "Event:",
            event.get(
                "event_id"
            ),
            "AI:",
            event.get(
                "status"
            ),
            "Human:",
            human_decision,
            "Final:",
            actual_final,
        )
        if actual_final != expected_final:
            failed = True

    if not items:
        print(
            "Published event store is empty; "
            "validating zero-state review statistics."
        )

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
        != expected_queue_total
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

    actual_final_counts = review_stats.get(
        "by_final_disposition",
        {},
    )
    for disposition in (
        "CONFIRMED_VIOLATION",
        "FALSE_ALARM",
        "NEEDS_REVIEW",
        "PENDING_REVIEW",
    ):
        if int(
            actual_final_counts.get(
                disposition,
                0,
            )
        ) != int(
            expected_final_counts.get(
                disposition,
                0,
            )
        ):
            failed = True

    if int(
        review_stats.get(
            "published_event_count",
            -1,
        )
    ) != len(items):
        failed = True

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
