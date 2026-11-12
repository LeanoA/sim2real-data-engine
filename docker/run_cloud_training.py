#!/usr/bin/env python3
"""Automated AWS EC2 GPU provisioning, Isaac Sim execution, and teardown manager.

Strictly adheres to AWS project constraints:
- Region: us-east-2 only
- Instance Type: g5.2xlarge (NVIDIA A10G 24GB VRAM)
- Automatic immediate termination upon completion to eliminate unnecessary spend
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REGION = os.environ.get("AWS_REGION", "us-east-2")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "592033927220")
PROFILE = os.environ.get("AWS_PROFILE", "xander-mac")
BUCKET = os.environ.get("SIM2REAL_S3_BUCKET", f"sim2real-artifacts-{ACCOUNT_ID}-{REGION}")
REQUEST_ID_ON_DEMAND = os.environ.get("AWS_QUOTA_REQ_ON_DEMAND", "7de4b3e954b94f61abb681416a5d4bc9tlC8PN1B")
REQUEST_ID_SPOT = os.environ.get("AWS_QUOTA_REQ_SPOT", "4489870f3a2d46d49ada8976a4681286M7YGU5Ao")
AMI_ID = "ami-032b37e4db407994d"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7
INSTANCE_TYPE = "g5.2xlarge"
IAM_PROFILE = "Sim2RealEC2Profile"


def run_aws(args: list[str]) -> dict | None:
    cmd = ["aws", *args, "--profile", PROFILE, "--region", REGION, "--output", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[AWS Error] {' '.join(cmd)}\n{res.stderr.strip()}", file=sys.stderr)
        return None
    out = res.stdout.strip()
    return json.loads(out) if out else {}


def check_quota_status(req_id: str) -> tuple[str, str]:
    data = run_aws(["service-quotas", "get-requested-service-quota-change", "--request-id", req_id])
    if not data or "RequestedQuota" not in data:
        return "UNKNOWN", ""
    q = data["RequestedQuota"]
    status = q.get("Status", "UNKNOWN")
    case_id = q.get("CaseId", "")
    return status, case_id


def build_user_data_script() -> str:
    """Startup bash script executed on instance boot."""
    return f"""#!/bin/bash
set -xe
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting Sim2Real Isaac Sim Cloud Job at $(date) ==="
export HOME=/root
export DEBIAN_FRONTEND=noninteractive

# Verify NVIDIA Driver & CUDA
nvidia-smi

# Prepare environment and extract code bundle from S3
mkdir -p /opt/sim2real-data-engine
cd /opt/sim2real-data-engine
aws s3 cp s3://{BUCKET}/code/sim2real-bundle.tar.gz /tmp/sim2real-bundle.tar.gz --region {REGION}
tar -xzf /tmp/sim2real-bundle.tar.gz -C /opt/sim2real-data-engine

# Install uv if missing
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Run Isaac Lab benchmarks
echo "Executing Isaac Lab sim2sim evaluation..."
uv run sim2real sim2sim-eval \\
  --ref mujoco_native \\
  --target isaac_lab \\
  --episodes 20 \\
  --json > /tmp/isaac_sim2sim_results.json || true

# Run policy training on NVIDIA A10G GPU
echo "Executing policy training on NVIDIA A10G (CUDA)..."
uv run sim2real train-policy \\
  --dataset-dir ./datasets/sim2real_pick_place \\
  --epochs 15 \\
  --batch-size 64 \\
  --device cuda \\
  --output-model /tmp/bc_cloud_policy.pt \\
  --json > /tmp/training_results.json || true

# Upload results to S3
aws s3 cp /tmp/isaac_sim2sim_results.json \\
  s3://{BUCKET}/results/isaac_sim2sim_results.json --region {REGION} || true
aws s3 cp /tmp/training_results.json \\
  s3://{BUCKET}/results/training_results.json --region {REGION} || true
aws s3 cp /tmp/bc_cloud_policy.pt \\
  s3://{BUCKET}/results/bc_cloud_policy.pt --region {REGION} || true
aws s3 cp /var/log/user-data.log s3://{BUCKET}/logs/user-data.log --region {REGION} || true

echo "=== Sim2Real Isaac Sim Cloud Job Completed at $(date) ==="
# Self-terminate or shutdown to prevent idle spend
shutdown -h now
"""


def launch_and_manage_instance(use_spot: bool = True) -> int:
    print(f"Launching {INSTANCE_TYPE} in {REGION} using AMI {AMI_ID} (Spot={use_spot})...")
    user_data_raw = build_user_data_script()
    user_data_b64 = base64.b64encode(user_data_raw.encode("utf-8")).decode("utf-8")

    launch_args = [
        "ec2", "run-instances",
        "--image-id", AMI_ID,
        "--instance-type", INSTANCE_TYPE,
        "--iam-instance-profile", f"Name={IAM_PROFILE}",
        "--user-data", user_data_b64,
        "--tag-specifications",
        "ResourceType=instance,Tags=[{Key=Name,Value=sim2real-isaac-worker}]",
    ]

    if use_spot:
        launch_args.extend(["--instance-market-options", json.dumps({"MarketType": "spot"})])

    launch_data = run_aws(launch_args)

    if not launch_data or "Instances" not in launch_data:
        if use_spot:
            print("Spot request failed or unavailable, attempting On-Demand...")
            return launch_and_manage_instance(use_spot=False)
        print("[Fatal] Could not launch instance.", file=sys.stderr)
        return 1

    instance_id = launch_data["Instances"][0]["InstanceId"]
    print(f"Instance launched successfully: {instance_id}")

    try:
        print(f"Monitoring instance {instance_id} until completion...")
        for _ in range(60):  # Wait up to 30 minutes
            time.sleep(30)
            status_data = run_aws(["ec2", "describe-instances", "--instance-ids", instance_id])
            if not status_data:
                continue
            inst = status_data["Reservations"][0]["Instances"][0]
            state = inst["State"]["Name"]
            print(f"Instance {instance_id} state: {state}")
            if state in ("stopped", "terminated"):
                print("Instance completed workload.")
                break
    finally:
        print(f"Terminating instance {instance_id} to eliminate all cloud spend...")
        term_data = run_aws(["ec2", "terminate-instances", "--instance-ids", instance_id])
        print(f"Termination result: {term_data}")

    # Fetch results and logs from S3
    print(f"Downloading results from s3://{BUCKET}/results/ ...")
    local_out = Path(__file__).resolve().parent.parent / "results"
    local_out.mkdir(exist_ok=True)
    subprocess.run([
        "aws", "s3", "cp", f"s3://{BUCKET}/results/", str(local_out),
        "--recursive", "--profile", PROFILE, "--region", REGION
    ])
    subprocess.run([
        "aws", "s3", "cp", f"s3://{BUCKET}/logs/user-data.log", str(local_out / "user-data.log"),
        "--profile", PROFILE, "--region", REGION
    ])
    return 0


def main() -> int:
    watch_mode = "--watch" in sys.argv
    force_on_demand = "--on-demand" in sys.argv
    force_spot = "--spot" in sys.argv

    while True:
        status_od, case_od = check_quota_status(REQUEST_ID_ON_DEMAND)
        status_spot, case_spot = check_quota_status(REQUEST_ID_SPOT)
        print(
            f"[{time.strftime('%X')}] Quotas -> "
            f"On-Demand: {status_od} (Case: {case_od}) | Spot: {status_spot} (Case: {case_spot})"
        )

        is_spot_ready = status_spot in ("APPROVED", "CASE_CLOSED")
        is_od_ready = status_od in ("APPROVED", "CASE_CLOSED")

        if force_spot and is_spot_ready:
            print("Launching GPU Spot workload...")
            return launch_and_manage_instance(use_spot=True)
        elif is_od_ready or force_on_demand:
            print("GPU Quota is granted! Launching On-Demand workload...")
            return launch_and_manage_instance(use_spot=False)
        elif is_spot_ready:
            print("GPU Spot Quota is granted! Launching Spot workload...")
            return launch_and_manage_instance(use_spot=True)

        if not watch_mode:
            print("Quota increases are still pending AWS review.")
            return 2

        print("Waiting 60 seconds before next check...")
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
