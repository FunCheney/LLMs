from ingestion.heading_detector import (
    Heading,
    SectionTracker,
)

tracker = SectionTracker()

tracker.update(
    Heading("1 Introduction", 1)
)
print(tracker.get_path())

tracker.update(
    Heading("1.1 Background", 2)
)
print(tracker.get_path())

tracker.update(
    Heading("1.2 Motivation", 2)
)
print(tracker.get_path())

tracker.update(
    Heading("2 Architecture", 1)
)
print(tracker.get_path())

def test_section_tracker():
    tracker = SectionTracker()

    tracker.update(
        Heading("1 Introduction", 1)
    )

    tracker.update(
        Heading("1.1 Background", 2)
    )

    assert tracker.get_path() == [
        "1 Introduction",
        "1.1 Background",
    ]

    tracker.update(
        Heading("1.2 Motivation", 2)
    )

    assert tracker.get_path() == [
        "1 Introduction",
        "1.2 Motivation",
    ]

    tracker.update(
        Heading("2 Architecture", 1)
    )

    assert tracker.get_path() == [
        "2 Architecture",
    ]