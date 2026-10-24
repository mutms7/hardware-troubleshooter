from troubleshooter.journal import Journal
from troubleshooter.service import MeasurementService
from troubleshooter.settings import Settings
from troubleshooter.stub import StubBackend


async def no_progress(stage, attempt):
    pass


async def test_ros_timeouts_unset_prevent_goal_dispatch(tmp_path):
    settings = Settings(backend="ros", journal=str(tmp_path / "operations.sqlite3"))
    backend = StubBackend(settings)

    def must_not_dispatch(*args):
        raise AssertionError("Unset ROS limits must prevent dispatch")

    backend.start = must_not_dispatch
    service = MeasurementService(settings, backend)
    try:
        result = await service.measure("TP_DEMO", "unset", no_progress)
        assert result.failure_reason == "configuration_required:timeouts"
        assert result.mode is None
        assert service.journal.get("unset") is None
    finally:
        await service.close()


async def test_unfinished_dispatch_blocks_restart(tmp_path):
    path = str(tmp_path / "operations.sqlite3")
    journal = Journal(path)
    journal.begin("interrupted", "TP_DEMO")
    journal.close()
    settings = Settings(backend="stub", journal=path)
    service = MeasurementService(settings, StubBackend(settings))
    try:
        result = await service.measure("TP_DEMO", "new-request", no_progress)
        assert result.failure_reason == "uncertain_motion_state"
        assert result.execution_uncertain
        stored = service.get_operation("interrupted")
        assert stored["execution_uncertain"]
        assert stored["reading"] is None
    finally:
        await service.close()
