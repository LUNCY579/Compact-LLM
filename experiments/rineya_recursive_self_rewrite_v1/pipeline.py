"""Budget-matched discovery and bounded planner/critic reconstruction."""
from __future__ import annotations

from dataclasses import asdict

from .core import (Backend, BudgetExceeded, FEEDBACK, Ledger, Limits, Runbook, STEPS,
                   Task, canonical, digest, loads, require, seed_for, validate_catalog)
from .runtime import audit, execute


def discovery(tasks: list[Task], backend: Backend, limits: Limits, *, attempts=2, seed=37) -> dict:
    validate_catalog(tasks)
    require(all(t.split == "train" for t in tasks), "discovery is train-only")
    require(type(attempts) is int and 1 <= attempts <= 16, "attempts")
    schedules = {"h0_resampling": ["H0"] * (3 * attempts),
                 "portfolio": [h for h in ("H0", "H1", "H2") for _ in range(attempts)]}
    records, ledgers = [], []
    for strategy, harnesses in schedules.items():
        for task in tasks:
            ledger = Ledger(limits)  # Identical total caps per task and strategy, not per harness.
            for slot, harness in enumerate(harnesses):
                records.append(execute(task, backend, ledger, harness=harness,
                                       arm=strategy, attempt=slot, seed=seed, run_id="discovery"))
            ledgers.append(dict(strategy=strategy, task=task.key,
                                report=ledger.report(), calls=ledger.events))
    return dict(records=records, ledgers=ledgers, schedules=schedules,
                task_keys=[t.key for t in tasks], limits=asdict(limits))


def discovery_report(result: dict) -> dict:
    rows = result["records"]
    expected = {(s, key, i) for s, hs in result["schedules"].items()
                for key in result["task_keys"] for i in range(len(hs))}
    actual = [(r["arm"], r["task"], r["attempt"]) for r in rows]
    require(len(set(actual)) == len(actual) and set(actual) == expected, "incomplete attempt coverage")
    for row in rows:
        require(row["harness"] == result["schedules"][row["arm"]][row["attempt"]], "schedule mismatch")
    out = {}
    for strategy in result["schedules"]:
        group = [r for r in rows if r["arm"] == strategy]
        out[strategy] = dict(attempts=len(group), successes=sum(r["success"] for r in group),
                             supported_successes=sum(r["success"] and r["grounded"] for r in group),
                             solved_tasks=len({r["task"] for r in group if r["success"]}),
                             failures={s: sum(r["status"] == s for r in group)
                                       for s in ("FAILURE", "TRUNCATED", "INVALID", "ERROR", "BUDGET")})
    sets = {h: {r["task"] for r in rows if r["arm"] == "portfolio" and r["harness"] == h and r["success"]}
            for h in ("H0", "H1", "H2")}
    out["exclusive_coverage"] = {h: len(values - set.union(*(s for other, s in sets.items() if other != h)))
                                 for h, values in sets.items()}
    for strategy in result["schedules"]:
        reports = [entry["report"] for entry in result["ledgers"] if entry["strategy"] == strategy]
        require(len(reports) == len(result["task_keys"]), "missing resource ledger")
        require(all(r["limits"] == result["limits"] for r in reports), "unequal resource caps")
        out[strategy]["actual_usage"] = {key: sum(r[key] for r in reports)
                                          for key in ("calls", "tools", "input_chars", "output_chars")}
        out[strategy]["actual_usage"]["tokens"] = (sum(r["tokens"] for r in reports)
                                                         if all(r["tokens"] is not None for r in reports) else None)
    out["budget_interpretation"] = "Equal attempt slots and caps; actual resource use may differ."
    return out


def reconstruct(task: Task, source: dict, executor: Backend, planner: Backend, critic: Backend,
                limits: Limits, *, arm="recursive", candidates=2, executions=2, revisions=2,
                seed=37, reviewer: Backend | None = None) -> dict:
    require(arm in {"one_pass", "recursive", "corrected"}, "rewrite arm")
    require(audit(source, task)["eligible"], "source is not replay/evidence qualified")
    require(source["source_id"] is None and source["arm"] in {"portfolio", "h0_resampling"}, "source lineage")
    require(source["backend"] == executor.identity and source["origin"] == executor.origin,
            "student-first executor mismatch")
    require((arm == "corrected") == (reviewer is not None), "correction must be separately labeled")
    for n in (candidates, executions):
        require(type(n) is int and 1 <= n <= 4, "rewrite multiplicity")
    require(type(revisions) is int and 0 <= revisions <= 4, "revision bound")
    rounds = 1 if arm == "one_pass" else revisions + 1
    ledger = Ledger(limits)
    drafts, records = [], []
    source_public = [{"action": s["action"], "observation": s["observation"]} for s in source["steps"]]
    for candidate in range(candidates):
        feedback, previous = "incomplete", None
        for revision in range(rounds):
            request = {"role": "planner", "task": task.prompt, "source_public": source_public,
                       "operations": STEPS, "feedback": feedback, "previous": previous,
                       "instruction": "Return only goal=evidence_backed_completion and steps (operation names). No literals."}
            entry = dict(candidate=candidate, revision=revision, source_id=source["id"],
                         source_hash=source["record_hash"], request=request, accepted=False)
            drafts.append(entry)
            try:
                raw = ledger.call(planner, "planner", canonical(request),
                                  seed_for(task.key, seed, candidate, revision, "planner"))
                entry["raw"] = raw
                try:
                    book = Runbook.parse(raw)
                except (ValueError, TypeError, KeyError):
                    feedback, previous = "schema", raw
                    entry["feedback"] = feedback
                    continue
                critique_request = {"role": "critic", "task": task.prompt,
                                    "runbook": loads(book.encode()), "operations": STEPS,
                                    "instruction": "Return request_hash, decision ACCEPT/REJECT, reason accepted/schema/unsupported/leakage/incomplete."}
                request_hash = digest(critique_request)
                critique_request["request_hash"] = request_hash
                critique = loads(ledger.call(critic, "critic", canonical(critique_request),
                                            seed_for(task.key, seed, candidate, revision, "critic")))
                entry["critique"] = critique
                require(type(critique) is dict and set(critique) == {"request_hash", "decision", "reason"}, "critic schema")
                require(critique["request_hash"] == request_hash, "stale critique")
                require(critique["decision"] in {"ACCEPT", "REJECT"} and critique["reason"] in FEEDBACK, "critic fields")
                require((critique["decision"] == "ACCEPT") == (critique["reason"] == "accepted"), "critic inconsistency")
                feedback, previous = critique["reason"], raw
                entry["feedback"] = feedback
                if critique["decision"] == "REJECT":
                    continue
                entry.update(accepted=True, runbook=book.encode(), runbook_hash=book.fingerprint)
                for trial in range(executions):
                    records.append(execute(task, executor, ledger, arm=arm,
                                           attempt=candidate * executions + trial, seed=seed,
                                           run_id="rewrite", runbook=book, source_id=source["id"], reviewer=reviewer))
                break
            except BudgetExceeded as exc:
                entry.update(feedback="budget", error=str(exc))
                break
            except Exception as exc:
                entry.update(feedback="error", error=type(exc).__name__)
                break  # Callback/contract failure is not a successful negative critique.
    helpers = [planner, critic] + ([reviewer] if reviewer else [])
    track = "self" if all(b.identity == executor.identity and b.origin == executor.origin for b in helpers) else "teacher_assisted"
    return dict(task=task.key, source_id=source["id"], arm=arm, track=track,
                helpers={"planner": {"identity": planner.identity, "origin": planner.origin},
                         "critic": {"identity": critic.identity, "origin": critic.origin},
                         "reviewer": {"identity": reviewer.identity, "origin": reviewer.origin} if reviewer else None},
                drafts=drafts, records=records, ledger=ledger.report(), calls=ledger.events)


def validate_rewrite(rewrite: dict, source: dict, task: Task) -> None:
    require(rewrite["source_id"] == source["id"] and rewrite["task"] == task.key, "rewrite ancestry")
    require(audit(source, task)["eligible"], "invalid source")
    expected_track = "self" if all(h["identity"] == source["backend"] and h["origin"] == source["origin"]
                                    for h in rewrite["helpers"].values() if h) else "teacher_assisted"
    require(rewrite["track"] == expected_track, "assistance provenance")
    accepted = {}
    for draft in rewrite["drafts"]:
        require(draft["source_id"] == source["id"] and draft["source_hash"] == source["record_hash"], "draft source")
        if draft["accepted"]:
            book = Runbook.parse(draft["runbook"])
            require(book.fingerprint == draft["runbook_hash"] and draft["feedback"] == "accepted", "draft approval")
            require(Runbook.parse(draft["raw"]).fingerprint == book.fingerprint, "draft payload drift")
            critique_request = {"role": "critic", "task": task.prompt,
                                "runbook": loads(book.encode()), "operations": STEPS,
                                "instruction": "Return request_hash, decision ACCEPT/REJECT, reason accepted/schema/unsupported/leakage/incomplete."}
            require(draft["critique"] == {"request_hash": digest(critique_request),
                                          "decision": "ACCEPT", "reason": "accepted"}, "critique replay mismatch")
            accepted[book.fingerprint] = book
    sandboxes = {source["sandbox"]}
    identities = set()
    for row in rewrite["records"]:
        require(row["id"] not in identities and row["sandbox"] not in sandboxes, "duplicate execution")
        identities.add(row["id"])
        sandboxes.add(row["sandbox"])
        require(row["source_id"] == source["id"] and row["arm"] == rewrite["arm"], "record lineage")
        require(row["backend"] == source["backend"] and row["harness"] == "H0", "executor contract")
        require(row["runbook"] in accepted, "unapproved runbook")
        book = accepted[row["runbook"]]
        if rewrite["arm"] == "corrected":
            require(row["review_reference"] == {"name": "rsr-procedures", "rules": tuple(STEPS[s] for s in book.steps)}
                    or row["review_reference"] == {"name": "rsr-procedures", "rules": [STEPS[s] for s in book.steps]},
                    "review reference drift")
        for step in row["steps"]:
            require(step["student_prompt"].endswith("\nPRIVATE_TRAINING_GUIDANCE\n" + book.guidance()), "guidance drift")
        audit(row, task)
