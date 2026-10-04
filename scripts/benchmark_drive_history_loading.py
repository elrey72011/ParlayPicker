"""Hermetic 39,000-object UI history benchmark; no private inputs or live clients."""
from contextlib import contextmanager
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import socket
import tempfile
from time import perf_counter
from types import SimpleNamespace
import urllib.request
import requests

from app_core.evidence_drive import API, DriveStore
from app_core.public_history import History, digest

BASE = "befe6a09cfcec3e6dbe01f53b92602828c3a0db7"
# Exact restore_history source at BASE; self-contained in shallow CI checkouts.
BASELINE_RESTORE_SOURCE = 'def restore_history(setting):\n    site=str(setting("PARLAYPICKER_NETLIFY_SITE_ID")).strip()\n    key="public_results_"+site\n    stage = \'opening history storage\'\n    try:\n        store=history(setting)\n        stage = \'recovering unconfirmed publications\'\n        # Recover known deployments after a Streamlit restart. Only the\n        # site\'s currently published deployment can be confirmed.\n        from app_core.netlify_publishing import deployment_status\n        token=str(setting(\'PARLAYPICKER_NETLIFY_TOKEN\')).strip()\n        for pending in store.all(\'deployments\'):\n            try:\n                store.read(\'confirmed/\'+pending[\'deploy_id\']+\'.json\')\n            except Exception:\n                status = None\n                try:\n                    if pending[\'deploy_id\'].startswith(\'sftp-\'):\n                        from app_core import sftp_publishing\n                        status=sftp_publishing.deployment_status(pending[\'deploy_id\'],sftp_publishing.configuration(setting))\n                    elif token:\n                        status=deployment_status(pending[\'deploy_id\'],site,token)\n                except (ValueError,RuntimeError):\n                    st.warning(\'An unconfirmed hosting publication could not be verified. Continuing to restore confirmed history; the unverified publication will not be counted.\')\n                if status and status[\'state\']==\'ready\':\n                    # Storage/integrity failures must still stop restore.\n                    store.confirm(pending[\'deploy_id\'],pending[\'package_hash\'])\n        stage = \'reading saved publications and results\'\n        pubs=store.publications();revisions=store.all(\'scores\');imports=store.all(\'imports\');locks=store.all(\'locks\')\n        st.session_state[key]={\'publications\':pubs,\'revisions\':revisions,\'imports\':imports,\'grading_runs\':store.all(\'grading_runs\'),\'prop_revisions\':store.all(\'prop_stats\'),\'prop_imports\':store.all(\'prop_imports\'),\'locks\':locks,\'rows\':report(pubs,revisions,imports,locks)}\n        st.session_state[\'relock_reset_requested\'] = True\n        st.success(\'Public history restored.\')\n        return True\n    except Exception as exc:\n        from app_core.evidence_config import safe_error\n        detail = safe_error(exc, \'History restore while \' + stage)\n        st.error(\'Public history restore failed while \' + stage + \'. \' + detail + \' No saved history was replaced.\')\n        return False\n'
BASELINE_RESTORE_SHA256 = "1625fef70d31ed7d89dfea8489a9ca5833db6dc493be6de18ff54845640b06be"


class Response:
    def __init__(self, value=None, content=b""):
        self.value, self.content = value, content
    def json(self):
        return self.value
    def raise_for_status(self):
        pass


class SyntheticDrive:
    def __init__(self, count=39000, principal="synthetic@example.invalid"):
        self.credentials = SimpleNamespace(service_account_email=principal)
        self.files = []
        self.listings = self.pages = self.downloads = 0
        self.incomplete = False
        package = dict(schema_version=5, built_at="2026-10-03T15:00:00+00:00",
            games={"overall": [], "sides": [], "totals": []}, props=[], dfs=[], results=[])
        prefix = "parlaypicker/public-history-v1/site-1234/"
        self.add(prefix+"packages/"+digest(package)+".json", package)
        self.add(prefix+"confirmed/deploy-1.json",
                 dict(package_hash=digest(package), confirmed_at="2026-10-03T15:01:00+00:00"))
        self.add(prefix+"deployments/deploy-1.json", dict(deploy_id="deploy-1", package_hash=digest(package)))
        self.add(prefix+"scores/one.json", dict(recorded_at="2026-10-03T16:00:00Z", scores=[]))
        self.add(prefix+"imports/one.json", dict(id="one", games=package["games"]))
        self.add(prefix+"grading_runs/one.json", dict(started_at="2026-10-03T16:00:00Z", status="complete"))
        self.add(prefix+"prop_stats/one.json", dict(recorded_at="2026-10-03T16:00:00Z", actuals=[]))
        self.add(prefix+"prop_imports/one.json", dict(id="one", as_of="2026-10-03T15:00:00+00:00", props=[]))
        self.files.extend(dict(id="filler-"+str(i), name="unrelated/"+str(i), content=b"")
                          for i in range(count-len(self.files)))

    def add(self, name, value, checksum=True):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        item = dict(id="object-"+str(len(self.files)), name=name, content=raw)
        if checksum:
            item["sha256Checksum"] = sha256(raw).hexdigest()
        self.files.append(item)
        return item

    def get(self, url, params, timeout, **kwargs):
        if params.get("alt") == "media":
            self.downloads += 1
            return Response(content=next(f["content"] for f in self.files if url.endswith("/"+f["id"])))
        if url != API:
            return Response(dict(id="folder", driveId="drive",
                mimeType="application/vnd.google-apps.folder", trashed=False))
        files = self.files
        query = params.get("q", "")
        if " and name = '" in query:
            name = query.split(" and name = '", 1)[1][:-1]
            files = [f for f in files if f["name"] == name]
        elif not params.get("pageToken"):
            self.listings += 1
        self.pages += 1
        offset = int(params.get("pageToken", 0))
        chunk = files[offset:offset+1000]
        value = dict(files=[{k:v for k,v in f.items() if k != "content"} for f in chunk],
                     incompleteSearch=self.incomplete)
        if offset+1000 < len(files):
            value["nextPageToken"] = str(offset+1000)
        return Response(value)


@contextmanager
def blocked_network():
    from unittest.mock import patch
    def blocked(*args, **kwargs):
        raise AssertionError("Network blocked in synthetic regression")
    with patch.object(socket, "create_connection", blocked), patch.object(socket.socket, "connect", blocked), \
         patch.object(socket, "getaddrinfo", blocked), patch.object(requests.sessions.Session, "request", blocked), \
         patch.object(urllib.request, "urlopen", blocked):
        yield


class UI:
    def __init__(self):
        self.session_state = {}
        self.errors = []
    def error(self, message):
        self.errors.append(message)
    def success(self, message):
        pass
    def warning(self, message):
        pass


def setting(name):
    return {"PARLAYPICKER_NETLIFY_SITE_ID":"site-1234",
            "PARLAYPICKER_DRIVE_FOLDER_ID":"folder"}.get(name, "")


def measure(operation, session):
    before = (session.listings, session.pages, session.downloads)
    start = perf_counter()
    operation()
    return dict(wall_seconds=round(perf_counter()-start, 6), inventories=session.listings-before[0],
        listing_requests=session.pages-before[1], downloads=session.downloads-before[2])


def benchmark(root):
    from app.ui import public_results
    previous_dir = os.environ.get("PARLAYPICKER_EVIDENCE_DIR")
    original_history, original_st = public_results.history, public_results.st
    os.environ["PARLAYPICKER_EVIDENCE_DIR"] = str(root / "before")
    try:
        from app_core.public_history import report
        assert sha256(BASELINE_RESTORE_SOURCE.encode()).hexdigest() == BASELINE_RESTORE_SHA256
        old = {"__name__":"synthetic_baseline", "report":report}
        exec(compile(BASELINE_RESTORE_SOURCE, "baseline_public_results.py", "exec"), old)
        cold_before = SyntheticDrive()
        legacy = History("site-1234", "folder", DriveStore("folder", session=cold_before))
        old["history"], old["st"] = lambda unused:legacy, UI()
        before = measure(lambda: old["restore_history"](setting), cold_before)
        # The original lock panel also read removals after restore.
        extra = measure(lambda: legacy.all("lock_removals"), cold_before)
        os.environ["PARLAYPICKER_EVIDENCE_DIR"] = str(root / "after")
        session = SyntheticDrive()
        store = History("site-1234", "folder", DriveStore("folder", session=session))
        public_results.history, public_results.st = lambda unused:store, UI()
        cold = measure(lambda: public_results.restore_history(setting), session)
        assert not public_results.st.errors
        cold["reused_objects"] = store.client.last_read_report.objects_reused
        warm = measure(lambda: public_results.restore_history(setting), session)
        warm["reused_objects"] = store.client.last_read_report.objects_reused
        full = measure(lambda: public_results.restore_history(setting, full_verification=True), session)
        assert not public_results.st.errors
        return dict(synthetic_objects=39000, baseline_ui_restore=before,
            baseline_lock_display=extra, batched_ui_restore=cold, warm_explicit_restore=warm,
            full_verification=full, timing_basis="Local synthetic wall time; no provider latency. Nested/overlapping span durations must not be summed.")
    finally:
        public_results.history, public_results.st = original_history, original_st
        if previous_dir is None:
            os.environ.pop("PARLAYPICKER_EVIDENCE_DIR", None)
        else:
            os.environ["PARLAYPICKER_EVIDENCE_DIR"] = previous_dir


if __name__ == "__main__":
    import sys
    logging.disable(logging.CRITICAL)
    with blocked_network(), tempfile.TemporaryDirectory() as directory:
        result = benchmark(Path(directory))
    print(json.dumps(result, indent=2))
