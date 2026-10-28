"""Local media and STM32 sEMG bridge. Python 3.10+; pyserial for board input."""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import queue
import threading
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from board_bridge import BoardBridge, DEFAULT_BAUDRATE

ROOT = Path(__file__).resolve().parent
WORDS = frozenset('open close next music start stop previous quieter louder photo'.split())
PHOTO_TYPES = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif', '.bmp'}
MUSIC_TYPES = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.opus'}
STATIC_FILES = {'index.html', 'styles.css', 'app.js', 'monitor.js', 'media-manifest.js'}


def media_manifest(root=ROOT):
    result = {}
    for folder, key, extensions in [('photo', 'photos', PHOTO_TYPES), ('music', 'music', MUSIC_TYPES)]:
        directory = root / folder
        files = sorted(directory.rglob('*'), key=lambda p: str(p).casefold()) if directory.is_dir() else []
        result[key] = [{'name': p.name, 'url': quote(p.relative_to(root).as_posix(), safe='/')}
                       for p in files if p.is_file() and p.suffix.lower() in extensions
                       and p.resolve().is_relative_to(root.resolve())]
    return result


def write_manifest():
    (ROOT / 'media-manifest.js').write_text(
        '// Generated locally by server.py.\nwindow.MEDIA_MANIFEST = ' +
        json.dumps(media_manifest(), ensure_ascii=False) + ';\n', encoding='utf-8')


def validate_command(data):
    if not isinstance(data, dict) or not isinstance(data.get('word'), str):
        raise ValueError('word must be one of the ten English commands')
    word = data['word'].strip().lower()
    if word not in WORDS:
        raise ValueError('unknown command')
    event = {'word': word, 'source': 'semg'}
    confidence = data.get('confidence')
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('confidence must be a finite number from 0 to 1')
        event['confidence'] = confidence
    event_id = data.get('id', str(uuid.uuid4()))
    if not isinstance(event_id, str) or not 1 <= len(event_id) <= 128:
        raise ValueError('id must be a string of 1 to 128 characters')
    event['id'] = event_id
    return event


def validate_signal(data):
    if not isinstance(data, dict):
        raise ValueError('signal must be a JSON object')
    channels = data.get('channels')
    if not isinstance(channels, list) or not 1 <= len(channels) <= 2:
        raise ValueError('channels must contain one or two arrays')
    for channel in channels:
        if not isinstance(channel, list) or not 2 <= len(channel) <= 500:
            raise ValueError('each channel needs 2 to 500 samples')
        if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in channel):
            raise ValueError('samples must be finite numbers')
    result = {'channels': channels}
    if 'sample_rate' in data:
        rate = data['sample_rate']
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('sample_rate must be positive')
        result['sample_rate'] = rate
    return result


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.seen = OrderedDict()

    def publish(self, kind, data):
        message = f'event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'.encode('utf-8')
        with self.lock:
            if kind == 'command' and data['id'] in self.seen:
                return {'ok': True, 'duplicate': True, 'subscribers': len(self.clients), 'id': data['id']}
            # Do not retain a command with no page listening: a retry can still deliver it.
            if kind == 'command' and self.clients:
                self.seen[data['id']] = True
                while len(self.seen) > 512:
                    self.seen.popitem(last=False)
            delivered = 0
            for client in list(self.clients):
                try:
                    client.put_nowait(message)
                    delivered += 1
                except queue.Full:
                    self.clients.discard(client)
            return {'ok': True, 'subscribers': delivered, **({'id': data['id']} if kind == 'command' else {})}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        self.hub = Hub()
        self.board = BoardBridge(self.hub)
        super().__init__(address, Handler)

    def server_close(self):
        self.board.disconnect()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def handle(self):
        # Browsers cancel keep-alive/media requests when navigating or seeking.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith('POST /api/signal'):
            return
        super().log_message(fmt, *args)

    def local_request(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host', '') not in allowed:
            self.discard_post_body()
            self.reply_json(403, {'ok': False, 'error': 'localhost access only'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{host}' for host in allowed}:
            self.discard_post_body()
            self.reply_json(403, {'ok': False, 'error': 'cross-origin requests are not allowed'})
            return False
        return True

    def discard_post_body(self):
        # Drain bounded rejected bodies before closing, avoiding Windows TCP RST.
        if self.command == 'POST':
            try:
                size = int(self.headers.get('Content-Length', '0'))
                self.connection.settimeout(1)
                if 0 < size <= 65536:
                    self.rfile.read(size)
            except (ValueError, OSError):
                pass

    def reply_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == '/api/media':
            self.reply_json(200, media_manifest())
        elif path == '/api/status':
            with self.server.hub.lock:
                count = len(self.server.hub.clients)
            self.reply_json(200, {'app': 'quietly', 'version': 3, 'subscribers': count})
        elif path == '/api/board/status':
            self.reply_json(200, self.server.board.snapshot())
        elif path == '/api/board/ports':
            try:
                self.reply_json(200, {'ports': self.server.board.ports()})
            except ValueError as exc:
                self.reply_json(400, {'error': str(exc)})
        elif path == '/api/events' and self.command == 'GET':
            self.events()
        else:
            self.serve_file(path)

    def events(self):
        client = queue.Queue(maxsize=256)
        hub = self.server.hub
        with hub.lock:
            hub.clients.add(client)
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(b'retry: 1500\n: connected\n\n')
            self.wfile.flush()
            while True:
                with hub.lock:
                    if client not in hub.clients:
                        break
                try:
                    message = client.get(timeout=2)
                except queue.Empty:
                    message = b': heartbeat\n\n'
                self.wfile.write(message)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with hub.lock:
                hub.clients.discard(client)
            self.close_connection = True

    def serve_file(self, raw_path):
        relative = unquote(raw_path).lstrip('/') or 'index.html'
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            self.reply_json(404, {'error': 'file not found'})
            return
        rel = path.relative_to(ROOT)
        is_media = (rel.parts[0] == 'photo' and path.suffix.lower() in PHOTO_TYPES) or (rel.parts[0] == 'music' and path.suffix.lower() in MUSIC_TYPES)
        if rel.as_posix() not in STATIC_FILES and not is_media:
            self.reply_json(404, {'error': 'file not found'})
            return
        size = path.stat().st_size
        start, end, partial = 0, size - 1, False
        requested_range = self.headers.get('Range')
        if requested_range:
            try:
                unit, interval = requested_range.split('=', 1)
                first, last = interval.split('-', 1)
                if unit != 'bytes' or ',' in interval or (not first and not last):
                    raise ValueError()
                if first:
                    start = int(first); end = min(int(last), size - 1) if last else size - 1
                else:
                    suffix = int(last)
                    if suffix <= 0:
                        raise ValueError()
                    start = max(0, size - suffix)
                if not 0 <= start <= end < size:
                    raise ValueError()
                partial = True
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        content_type = {'.js': 'text/javascript', '.css': 'text/css', '.mp3': 'audio/mpeg'}.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command == 'HEAD':
            return
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if not self.local_request():
            self.close_connection = True
            return
        endpoint = urlsplit(self.path).path
        if endpoint not in {'/api/command', '/api/signal', '/api/board/connect', '/api/board/disconnect', '/api/board/control', '/api/board/scan-ble'}:
            self.reply_json(404, {'ok': False, 'error': 'unknown endpoint'})
            self.close_connection = True
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536:
                raise ValueError('request body must be 1 to 65536 bytes')
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                raise ValueError('Content-Type must be application/json')
            self.connection.settimeout(5)
            data = json.loads(self.rfile.read(size))
            if endpoint.startswith('/api/board/'):
                if not isinstance(data, dict):
                    raise ValueError('request must be a JSON object')
                if endpoint.endswith('/connect'):
                    if 'transport' in data:
                        reply = self.server.board.connect(data.get('port'), data.get('baudrate', DEFAULT_BAUDRATE), data['transport'])
                    else:
                        reply = self.server.board.connect(data.get('port'), data.get('baudrate', DEFAULT_BAUDRATE))
                elif endpoint.endswith('/scan-ble'):
                    reply = {'ports': self.server.board.scan_ble()}
                elif endpoint.endswith('/disconnect'):
                    reply = self.server.board.disconnect()
                else:
                    reply = self.server.board.enable(data.get('enabled'))
                self.reply_json(200, reply)
                return
            kind = 'command' if endpoint == '/api/command' else 'signal'
            event = validate_command(data) if kind == 'command' else validate_signal(data)
            self.reply_json(200, self.server.hub.publish(kind, event))
        except (ValueError, UnicodeDecodeError, TimeoutError) as error:
            self.reply_json(400, {'ok': False, 'error': str(error)})
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Open the default browser')
    parser.add_argument('--manifest-only', action='store_true', help='Refresh the file:// fallback manifest')
    args = parser.parse_args()
    write_manifest()
    if args.manifest_only:
        print('Media manifest refreshed.')
        return
    try:
        server = LocalServer(('127.0.0.1', args.port))
    except OSError as error:
        print(f'Cannot listen on port {args.port}: {error}\nTry: python server.py --port 8766 --open', flush=True)
        raise SystemExit(1)
    url = f'http://127.0.0.1:{args.port}'
    print(f'Quietly is ready: {url}\nCommand input: POST {url}/api/command\nPress Ctrl+C to stop.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
