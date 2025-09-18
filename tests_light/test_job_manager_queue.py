import asyncio


def test_job_manager_worker_runs(monkeypatch):
    from services.api.jobs import JobManager

    async def scenario():
        mgr = JobManager()
        mgr.start()

        done = asyncio.Event()

        async def fake_run(jid: str):
            done.set()

        monkeypatch.setattr(mgr, "run", fake_run)

        await mgr.enqueue("job-1")
        await asyncio.wait_for(done.wait(), timeout=1)
        await mgr.stop()

    asyncio.run(scenario())
