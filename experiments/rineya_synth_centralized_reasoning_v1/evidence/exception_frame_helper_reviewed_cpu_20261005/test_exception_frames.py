"""Independent CPU controls for original traceback metadata, never native proof."""
import hashlib
import json
import unittest

from exception_frames import capture_exception_frames


PIN = "a" * 64
OTHER_PIN = "b" * 64
FILENAME = "C:\\synthetic_fixture\\registered_source.py"
LIMITS = {"maximum_frames": 16, "maximum_filename_chars": 512,
          "maximum_function_chars": 128, "maximum_serialized_bytes": 16384}
TRACEBACK = BaseException.__dict__["__traceback__"]
FRAME_FIELDS = {"filename", "function", "line", "registration_domains",
                "registered_sha256", "registration_status"}


def serialized(packet):
    return json.dumps(packet, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def traceback_records(error):
    result = []
    current = TRACEBACK.__get__(error, BaseException)
    while current is not None:
        result.append((current, current.tb_frame, current.tb_frame.f_code,
                       current.tb_lineno, current.tb_lasti))
        current = current.tb_next
    return result


def fixture_error(code=None, filename=FILENAME, exception_type=RuntimeError, depth=None):
    """Compile owned tiny CPU fixtures with literal filenames; never open a file."""
    namespace = {"FixtureException": exception_type}
    if code is None:
        code = ("def leaf():\n"
                "    secret_local = 'PRIVATE_LOCAL_SENTINEL'\n"
                "    raise FixtureException('PRIVATE_MESSAGE_SENTINEL')\n"
                "def middle():\n"
                "    leaf()\n"
                "def outer():\n"
                "    middle()\n")
    exec(compile(code, filename, "exec"), namespace)
    try:
        namespace["outer"]() if depth is None else namespace["outer"](depth)
    except BaseException as error:
        return error
    raise AssertionError("fixture must raise")


class ExceptionFrameControls(unittest.TestCase):
    def assert_sealed(self, packet, limit=16384):
        body = {key: value for key, value in packet.items() if key != "sha256"}
        self.assertEqual(packet["sha256"], hashlib.sha256(serialized(body)).hexdigest())
        self.assertLessEqual(len(serialized(packet)), limit)
        self.assertEqual(packet["frames_retained"], len(packet["frames"]))
        for frame in packet["frames"]:
            self.assertEqual(set(frame), FRAME_FIELDS)

    def test_original_head_order_and_exact_registered_line_metadata(self):
        error = fixture_error()
        original = traceback_records(error)
        pins = {record[2].co_filename: PIN for record in original}
        packet = capture_exception_frames(error, pins, {})
        self.assertEqual(packet["status"], "CAPTURED")
        expected = [(code.co_filename, code.co_name, line) for _, _, code, line, _ in original]
        actual = [(frame["filename"], frame["function"], frame["line"]) for frame in packet["frames"]]
        self.assertEqual(actual, expected)
        self.assertEqual([name for filename, name, _ in actual if filename == FILENAME],
                         ["outer", "middle", "leaf"])
        self.assert_sealed(packet)

    def test_chained_exception_reads_only_current_traceback(self):
        for suffix in (" from cause", ""):
            error = fixture_error(
                "def cause_leaf():\n"
                "    raise ZeroDivisionError('PRIVATE_CAUSE_SENTINEL')\n"
                "def outer():\n"
                "    try:\n"
                "        cause_leaf()\n"
                "    except ZeroDivisionError as cause:\n"
                "        raise RuntimeError('PRIVATE_CURRENT_SENTINEL')" + suffix + "\n")
            with self.subTest(explicit_cause=bool(suffix)):
                packet = capture_exception_frames(error, {FILENAME: PIN}, {})
                names = [frame["function"] for frame in packet["frames"]]
                self.assertIn("outer", names)
                self.assertNotIn("cause_leaf", names)
                self.assertNotIn(b"PRIVATE_CAUSE_SENTINEL", serialized(packet))
                self.assertNotIn(b"PRIVATE_CURRENT_SENTINEL", serialized(packet))
                self.assertFalse(packet["cause_established"])

    def test_hostile_exception_strings_properties_and_attribute_hooks_are_never_called(self):
        attempts = []
        class Poisoned(Exception):
            def __getattribute__(self, name):
                attempts.append("attribute:" + name)
                raise AssertionError("exception attribute hook invoked")
            def __str__(self):
                attempts.append("str")
                raise AssertionError("exception str invoked")
            def __repr__(self):
                attempts.append("repr")
                raise AssertionError("exception repr invoked")
            @property
            def __traceback__(self):
                attempts.append("traceback property")
                raise AssertionError("traceback property invoked")
            @property
            def args(self):
                attempts.append("args property")
                raise AssertionError("args property invoked")
        error = fixture_error(exception_type=Poisoned)
        original = TRACEBACK.__get__(error, BaseException)
        attempts.clear()
        packet = capture_exception_frames(error, {FILENAME: PIN}, {})
        self.assertEqual(attempts, [])
        self.assertIs(TRACEBACK.__get__(error, BaseException), original)
        self.assertEqual(packet["status"], "CAPTURED")
        self.assert_sealed(packet)

    def test_literal_filename_join_does_not_normalize_or_expose_unregistered_names(self):
        filename = "C:\\synthetic_fixture\\private_component_name.py"
        error = fixture_error(filename=filename)
        packet = capture_exception_frames(error, {filename.lower(): PIN}, {})
        self.assertTrue(packet["frames"])
        for frame in packet["frames"]:
            self.assertEqual(frame["filename"], "<unregistered>")
            self.assertEqual(frame["function"], "<unregistered>")
            self.assertEqual(frame["registration_status"], "unregistered_redacted")
            self.assertEqual(frame["registration_domains"], [])
            self.assertIsNone(frame["registered_sha256"])
        self.assertNotIn(b"private_component_name", serialized(packet))

    def test_source_runtime_same_pin_join_and_conflicting_pin_rejection(self):
        error = fixture_error()
        packet = capture_exception_frames(error, {FILENAME: PIN}, {FILENAME: PIN})
        registered = [frame for frame in packet["frames"] if frame["filename"] == FILENAME]
        self.assertEqual(len(registered), 3)
        for frame in registered:
            self.assertEqual(frame["registration_domains"], ["source", "runtime"])
            self.assertEqual(frame["registered_sha256"], PIN)
        with self.assertRaises(ValueError):
            capture_exception_frames(error, {FILENAME: PIN}, {FILENAME: OTHER_PIN})

    def test_pin_formats_and_non_exception_inputs_reject_before_capture(self):
        error = fixture_error()
        bad_sources = [[], {FILENAME: "A" * 64}, {FILENAME: "a" * 63},
                       {FILENAME: True}, {"": PIN}, {12: PIN}, {FILENAME: "g" * 64}]
        for sources in bad_sources:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                capture_exception_frames(error, sources, {})
        with self.assertRaises(ValueError):
            capture_exception_frames(object(), {}, {})

    def test_registered_oversized_names_use_fixed_redaction_markers(self):
        function = "f" * 129
        filename = "synthetic_" + "x" * 513
        error = fixture_error("def " + function + "():\n    raise RuntimeError('private')\n"
                              "def outer():\n    " + function + "()\n", filename=filename)
        packet = capture_exception_frames(error, {filename: PIN}, {})
        registered = [frame for frame in packet["frames"] if frame["registered_sha256"] == PIN]
        self.assertEqual([frame["filename"] for frame in registered], ["<name-limit>"] * 2)
        self.assertEqual(registered[-1]["function"], "<name-limit>")
        self.assertTrue(all(frame["registration_status"] == "name_limit_redacted" for frame in registered))
        self.assertNotIn(filename.encode(), serialized(packet))
        self.assertNotIn(function.encode(), serialized(packet))
        self.assert_sealed(packet)

    def test_frame_limit_retains_original_prefix_and_marks_truncation(self):
        error = fixture_error("def outer(depth):\n"
                              "    if depth:\n        outer(depth - 1)\n"
                              "    else:\n        raise RuntimeError('private')\n", depth=24)
        original = traceback_records(error)
        pins = {record[2].co_filename: PIN for record in original}
        limits = {**LIMITS, "maximum_frames": 4}
        packet = capture_exception_frames(error, pins, {}, limits=limits)
        self.assertEqual(packet["status"], "TRUNCATED_FRAME_LIMIT")
        self.assertEqual(packet["truncation_reason"], "frame_limit")
        self.assertIs(packet["truncated"], True)
        self.assertEqual([(frame["filename"], frame["function"], frame["line"]) for frame in packet["frames"]],
                         [(code.co_filename, code.co_name, line) for _, _, code, line, _ in original[:4]])
        self.assert_sealed(packet)

    def test_serialized_bound_produces_fixed_small_metadata_fallback(self):
        filename = "synthetic_" + "x" * 500
        function = "f" * 128
        error = fixture_error("def " + function + "():\n    raise RuntimeError('private')\n"
                              "def outer():\n    " + function + "()\n", filename=filename)
        # Select the synthetic traceback tail before handing the error to the
        # helper so its first registered frame alone exceeds the 1024-byte cap.
        tail = TRACEBACK.__get__(error, BaseException).tb_next
        TRACEBACK.__set__(error, tail)
        limits = {**LIMITS, "maximum_serialized_bytes": 1024}
        packet = capture_exception_frames(error, {filename: PIN}, {}, limits=limits)
        self.assertEqual(packet["status"], "TRUNCATED_SIZE_LIMIT")
        self.assertEqual(packet["frames"], [])
        self.assertEqual(packet["frames_retained"], 0)
        self.assertEqual(packet["truncation_reason"], "serialized_size")
        self.assertIs(packet["truncated"], True)
        self.assertIs(TRACEBACK.__get__(error, BaseException), tail)
        self.assert_sealed(packet, 1024)

    def test_error_identity_and_every_original_traceback_frame_remain_unchanged(self):
        error = fixture_error()
        identity = id(error)
        before = traceback_records(error)
        capture_exception_frames(error, {FILENAME: PIN}, {})
        after = traceback_records(error)
        self.assertEqual(id(error), identity)
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            self.assertIs(old[0], new[0])
            self.assertIs(old[1], new[1])
            self.assertIs(old[2], new[2])
            self.assertEqual(old[3:], new[3:])

    def test_no_traceback_packet_is_self_sealed_and_all_confidence_flags_stay_closed(self):
        packet = capture_exception_frames(RuntimeError("PRIVATE_MESSAGE_SENTINEL"), {}, {})
        self.assertEqual(packet["status"], "NO_TRACEBACK")
        self.assertEqual(packet["frames"], [])
        self.assertIs(packet["pins_are_supplied_not_byte_verified"], True)
        self.assertIs(packet["exception_preserved"], True)
        for name in ("locals_args_repr_contexts_tensors_read", "cause_established",
                     "native_qualification_established", "scoring_authorized"):
            self.assertIs(packet[name], False)
        self.assertNotIn(b"PRIVATE_MESSAGE_SENTINEL", serialized(packet))
        self.assert_sealed(packet)

    def test_limit_schema_types_and_fixed_upper_bounds_cannot_be_relaxed(self):
        bad_limits = [{}, {**LIMITS, "unknown": 1}, {**LIMITS, "maximum_frames": True},
                      {**LIMITS, "maximum_frames": 17}, {**LIMITS, "maximum_filename_chars": 513},
                      {**LIMITS, "maximum_function_chars": 129},
                      {**LIMITS, "maximum_serialized_bytes": 1023},
                      {**LIMITS, "maximum_serialized_bytes": 16385}]
        for limits in bad_limits:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                capture_exception_frames(RuntimeError(), {}, {}, limits=limits)


if __name__ == "__main__":
    unittest.main()
