def test_set_deterministic_flags():
    # Import function via JobManager static method pathway
    from services.api.jobs import JobManager  # type: ignore

    # Should not raise and should be callable with or without torch/cuda
    JobManager._set_deterministic(seed=123, deterministic=True)
    # Second call with None to exercise branches
    JobManager._set_deterministic(seed=None, deterministic=False)
