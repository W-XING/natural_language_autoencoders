#!/usr/bin/env python3.11
"""Auto-grab poller for an 8xH100 RL test pod.

Strategy: repeatedly attempt to create the REAL US-NE-1 8x pod with the existing
volume 2e7ynkygr5 mounted (zero transfer needed — checkpoints/venv/miles/repo/parquet
are all already on that volume). Creating the real pod directly (not a probe) avoids
a terminate->recreate gap where capacity could be lost. Also logs US-GA-2 8x stock so
we can see if that opens up (US-GA-2 would need a fresh volume + ~93GB transfer, so it
is NOT auto-grabbed here). Exits 0 on success after writing the grabbed pod id.
"""
import json, sys, time, traceback, urllib.request, urllib.error, pathlib

ENV = "/root/natural_language_autoencoders/.env"
KEY = [l.split("=",1)[1].strip().strip('"') for l in open(ENV) if l.strip().startswith("RUNPOD_API_KEY=")][0]
PUBKEY = pathlib.Path("/root/.ssh/id_ed25519.pub").read_text().strip()
HDR = {"authorization": f"Bearer {KEY}", "user-agent": "runpodctl/1.14.4", "content-type": "application/json"}
RESULT = "/tmp/rl8x_grabbed.txt"

INTERVAL = 240          # seconds between attempts
MAX_ATTEMPTS = 50       # ~3.3 hours

def rest(path, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"https://rest.runpod.io/v1{path}", data=data, method=method, headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, (json.load(r) if r.status != 204 else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception:
        traceback.print_exc()
        return -1, "exception"

def gql_stock(count, dc):
    q = '{ gpuTypes(input:{id:"NVIDIA H100 80GB HBM3"}){ lowestPrice(input:{gpuCount:%d, dataCenterId:"%s"}){ stockStatus } } }' % (count, dc)
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["data"]["gpuTypes"][0]["lowestPrice"].get("stockStatus")
    except Exception:
        return "err"

def try_grab_usne1():
    body = {"name": "nla-rl-8x", "imageName": "runpod/base:0.7.0-ubuntu2004", "computeType": "GPU",
            "cloudType": "SECURE", "gpuTypeIds": ["NVIDIA H100 80GB HBM3"], "gpuCount": 8,
            "containerDiskInGb": 60, "dataCenterIds": ["US-NE-1"], "networkVolumeId": "2e7ynkygr5",
            "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": PUBKEY}}
    st, resp = rest("/pods", "POST", body)
    if st in (200, 201) and isinstance(resp, dict) and resp.get("id"):
        return resp
    return None

for attempt in range(1, MAX_ATTEMPTS + 1):
    grabbed = try_grab_usne1()
    if grabbed:
        line = f"{grabbed['id']} US-NE-1 2e7ynkygr5 machine={grabbed.get('machineId')} cost={grabbed.get('costPerHr')}"
        pathlib.Path(RESULT).write_text(line)
        print(f"[attempt {attempt}] GRABBED US-NE-1 8x: {line}", flush=True)
        sys.exit(0)
    usga2 = gql_stock(8, "US-GA-2")
    print(f"[attempt {attempt}/{MAX_ATTEMPTS}] US-NE-1 8x not available; US-GA-2 8x stock={usga2}", flush=True)
    if attempt < MAX_ATTEMPTS:
        time.sleep(INTERVAL)

print("[poller] exhausted attempts without grabbing 8x", flush=True)
sys.exit(2)
