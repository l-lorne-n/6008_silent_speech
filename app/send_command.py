"""Small client usable from the existing upper-computer inference callback."""
import argparse
import json
import uuid
from urllib.request import Request, urlopen


def send_command(word, confidence=None, *, event_id=None, port=8765):
    """Send one completed prediction, using its class name (never a class index)."""
    payload = {'word': word, 'id': event_id or str(uuid.uuid4())}
    if confidence is not None:
        payload['confidence'] = float(confidence)
    request = Request(f'http://127.0.0.1:{port}/api/command',
                      data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=2) as response:
        return json.load(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('word')
    parser.add_argument('--confidence', type=float)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(json.dumps(send_command(args.word, args.confidence, port=args.port), ensure_ascii=False))
