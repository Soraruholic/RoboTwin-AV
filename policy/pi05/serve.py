"""Serve a fine-tuned AV checkpoint through the OpenPI websocket protocol."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from policy.pi05.config import register
    policy = policy_config.create_trained_policy(register(), args.checkpoint)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata={
        'action_type': 'dense_joint_position_velocity', 'action_dim': 28,
        'state_dim': 16, 'control_hz': 250,
    }).serve_forever()


if __name__ == '__main__':
    main()
