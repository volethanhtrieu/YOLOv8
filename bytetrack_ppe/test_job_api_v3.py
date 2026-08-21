from app import create_app


def main():
    app = create_app()
    app.config.update(
        TESTING=True
    )

    client = app.test_client()

    failed = False

    basic = [
        "/api/health",
        "/api/events",
        "/api/stats",
        "/api/jobs",
    ]

    for url in basic:
        response = client.get(
            url
        )

        print(
            response.status_code,
            url,
        )

        if response.status_code != 200:
            failed = True

    jobs_response = client.get(
        "/api/jobs"
    )

    jobs = (
        jobs_response.get_json()
        .get(
            "items",
            [],
        )
    )

    completed = next(
        (
            job
            for job in jobs
            if job.get(
                "status"
            )
            == "COMPLETED"
        ),
        None,
    )

    if completed is not None:
        job_id = completed[
            "job_id"
        ]

        for url in (
            (
                f"/api/jobs/{job_id}"
                "/preview/events"
            ),
            (
                f"/api/jobs/{job_id}"
                "/preview/video"
            ),
        ):
            response = client.get(
                url
            )

            print(
                response.status_code,
                url,
                response.content_type,
            )

            if response.status_code != 200:
                failed = True

    if failed:
        raise SystemExit(
            "Job Manager V3 smoke "
            "test failed."
        )

    print()
    print(
        "ALL JOB MANAGER V3 "
        "SMOKE TESTS PASSED"
    )


if __name__ == "__main__":
    main()
