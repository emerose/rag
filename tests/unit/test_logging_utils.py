import rag.utils.logging_utils as logging_utils


def test_log_message_passes_task_id() -> None:
    received: list[str | None] = []

    def cb(level: str, message: str, subsystem: str, task_id: str | None) -> None:
        received.append(task_id)

    logging_utils.log_message("INFO", "msg", "Test", cb, task_id="42")
    assert received[0] == "42"

