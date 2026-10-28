"""BT24 BLE receive check for the six-column CSV firmware in this repository.

This tool never sends AT commands or changes the module's configuration.
It displays received data; it is not the formal full-rate acquisition app.
"""
import argparse
import asyncio
from collections import deque
from pathlib import Path
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / ".deps"))

SERVICE = "0000ffe0-0000-1000-8000-00805f9b34fb"
NOTIFY = "0000ffe1-0000-1000-8000-00805f9b34fb"


class CsvReceiver:
    """A BLE notification may contain part of a line or many whole lines."""

    def __init__(self):
        self.buffer = bytearray()
        self.bytes = 0
        self.lines = 0
        self.bad_lines = 0
        self.synced = False
        self.latest = None
        self.preview = deque(maxlen=12)

    def feed(self, data):
        self.bytes += len(data)
        self.buffer.extend(data)
        while b"\n" in self.buffer:
            line, _, self.buffer = self.buffer.partition(b"\n")
            # Connection can start in the middle of a continuously sent line.
            if not self.synced:
                self.synced = True
                continue
            text = line.decode("ascii", errors="replace").strip()
            if not text:
                continue
            self.preview.append(text[:160])
            try:
                values = tuple(int(v) for v in text.split(","))
                if len(values) != 6 or not all(0 <= v <= 65535 for v in values[:2]):
                    raise ValueError("not a six-column ADC line")
            except ValueError:
                self.bad_lines += 1
                continue
            self.lines += 1
            self.latest = values
        if len(self.buffer) > 4096:
            self.buffer.clear()
            self.synced = False
            self.bad_lines += 1


def is_candidate(device, advert):
    name = advert.local_name or device.name or ""
    return ("BT24" in name.upper()
            or SERVICE in [s.lower() for s in advert.service_uuids])


class BleWorker:
    def __init__(self):
        self.events = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.devices = {}
        self.session = None
        self.stop_event = None
        self.receiver = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, operation):
        future = asyncio.run_coroutine_threadsafe(operation, self.loop)

        def finished(result):
            try:
                result.result()
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        future.add_done_callback(finished)

    async def scan(self):
        from bleak import BleakScanner
        found = await BleakScanner.discover(timeout=8, return_adv=True)
        self.devices = {d.address: d for d, a in found.values() if is_candidate(d, a)}
        rows = [(d.address, a.local_name or d.name or "BT24", a.rssi)
                for d, a in found.values() if is_candidate(d, a)]
        rows.sort(key=lambda row: ("BT24" not in row[1].upper(), -row[2]))
        self.events.put(("scan", (rows, len(found))))

    async def connect(self, address):
        from bleak import BleakClient
        if self.session is not None:
            raise RuntimeError("已有连接正在进行，请先断开")
        self.session = asyncio.current_task()
        self.stop_event = asyncio.Event()
        self.receiver = CsvReceiver()
        disconnected = asyncio.Event()
        started = None
        try:
            async with BleakClient(
                self.devices[address], pair=False,
                disconnected_callback=lambda _: disconnected.set(),
                winrt={"use_cached_services": False},
            ) as client:
                candidates = [c for s in client.services for c in s.characteristics
                              if s.uuid.lower() == SERVICE and "notify" in c.properties]
                characteristic = next((c for c in candidates if c.uuid.lower() == NOTIFY), None)
                if characteristic is None and len(candidates) == 1:
                    characteristic = candidates[0]
                if characteristic is None:
                    raise RuntimeError("未发现 BT24 的 FFE0/FFE1 通知通道，请核对模块设置")
                await client.start_notify(characteristic, lambda _, data: self.receiver.feed(data))
                started = time.monotonic()
                self.events.put(("connected", address))
                previous_time, previous_bytes, previous_lines = started, 0, 0
                while not self.stop_event.is_set() and not disconnected.is_set():
                    await asyncio.sleep(.25)
                    now = time.monotonic()
                    receiver = self.receiver
                    elapsed = now - previous_time
                    self.events.put(("data", dict(
                        bytes=receiver.bytes, lines=receiver.lines, bad=receiver.bad_lines,
                        byte_rate=round((receiver.bytes - previous_bytes) / elapsed),
                        line_rate=round((receiver.lines - previous_lines) / elapsed),
                        latest=receiver.latest, preview=list(receiver.preview),
                        seconds=round(now - started),
                    )))
                    previous_time, previous_bytes, previous_lines = now, receiver.bytes, receiver.lines
                if disconnected.is_set() and not self.stop_event.is_set():
                    self.events.put(("error", "蓝牙连接已意外断开；请检查供电和距离，然后重新扫描连接。"))
        finally:
            self.session = None
            self.stop_event = None
            self.events.put(("disconnected", None))

    async def disconnect(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.session is not None:
            await asyncio.wait_for(asyncio.shield(self.session), timeout=5)

    async def shutdown(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.session is not None:
            self.session.cancel()
        other = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in other:
            task.cancel()
        await asyncio.gather(*other, return_exceptions=True)


def run_gui():
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("BT24 蓝牙接收检查")
    root.geometry("870x650")
    root.minsize(750, 570)
    style = ttk.Style(root)
    style.configure("TLabel", font=("Microsoft YaHei UI", 10))
    style.configure("TButton", font=("Microsoft YaHei UI", 10))
    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="BT24 蓝牙接收检查", font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w")
    ttk.Label(frame, text="先关闭手机蓝牙，给设备上电；无需在 Windows 设置中配对。").pack(anchor="w", pady=(7, 12))
    worker = BleWorker()
    toolbar = ttk.Frame(frame)
    toolbar.pack(fill="x")
    status = tk.StringVar(value="准备就绪：点击“扫描 BT24”。")
    stats = tk.StringVar(value="累计接收 0 字节   |   有效数据 0 行")
    latest = tk.StringVar(value="等待六列传感器数据：原始 CH0、原始 CH1、滤波 CH0、滤波 CH1、包络 CH0、包络 CH1")
    connected, closing = False, False

    def scan():
        scan_button.configure(state="disabled")
        connect_button.configure(state="disabled")
        tree.delete(*tree.get_children())
        status.set("正在扫描，请等待 8 秒……")
        worker.submit(worker.scan())

    def connect():
        selection = tree.selection()
        if not selection:
            status.set("请先扫描并选择列表中的 BT24。")
            return
        scan_button.configure(state="disabled")
        connect_button.configure(state="disabled")
        status.set("正在连接并开启数据接收，请稍候……")
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")
        worker.submit(worker.connect(selection[0]))

    def disconnect():
        disconnect_button.configure(state="disabled")
        status.set("正在断开……")
        worker.submit(worker.disconnect())

    scan_button = ttk.Button(toolbar, text="① 扫描 BT24", command=scan)
    scan_button.pack(side="left")
    connect_button = ttk.Button(toolbar, text="② 连接选中设备", command=connect, state="disabled")
    connect_button.pack(side="left", padx=10)
    disconnect_button = ttk.Button(toolbar, text="断开", command=disconnect, state="disabled")
    disconnect_button.pack(side="left")
    tree = ttk.Treeview(frame, columns=("name", "address", "rssi"), show="headings", height=4)
    for column, title, width in (("name", "设备名称", 220), ("address", "地址", 300), ("rssi", "信号强度", 100)):
        tree.heading(column, text=title)
        tree.column(column, width=width)
    tree.pack(fill="x", pady=12)
    ttk.Label(frame, textvariable=status, wraplength=800).pack(anchor="w", pady=(0, 8))
    ttk.Label(frame, textvariable=stats).pack(anchor="w", pady=(0, 8))
    ttk.Label(frame, textvariable=latest, wraplength=800).pack(anchor="w", pady=(0, 10))
    ttk.Label(frame, text="最近收到的内容（滚动显示）：").pack(anchor="w")
    log = ScrolledText(frame, height=12, font=("Consolas", 11), state="disabled")
    log.pack(fill="both", expand=True, pady=(5, 10))
    ttk.Label(frame, text="本工具只验证电脑接收；正式上位机的波形、录制和事件标记尚未接入。", wraplength=800).pack(anchor="w")

    def poll():
        nonlocal connected
        if closing:
            return
        try:
            while True:
                kind, value = worker.events.get_nowait()
                if kind == "scan":
                    rows, count = value
                    for address, name, rssi in rows:
                        tree.insert("", "end", iid=address, values=(name, address, f"{rssi} dBm"))
                    scan_button.configure(state="normal")
                    if rows:
                        tree.selection_set(rows[0][0])
                        connect_button.configure(state="normal")
                        status.set(f"找到 {len(rows)} 个候选设备，请选择你的 BT24 并连接。")
                    else:
                        status.set(f"本轮扫描发现 {count} 个 BLE 设备，但没有 BT24。请关闭手机蓝牙，给 BT24 重新上电后再扫描。")
                elif kind == "connected":
                    connected = True
                    disconnect_button.configure(state="normal")
                    status.set("已连接 BT24，等待传感器数据……")
                elif kind == "disconnected":
                    connected = False
                    scan_button.configure(state="normal")
                    connect_button.configure(state="normal" if tree.get_children() else "disabled")
                    disconnect_button.configure(state="disabled")
                    if not status.get().startswith("出错"):
                        status.set("连接已断开。可重新扫描、连接。")
                elif kind == "error":
                    status.set("出错：" + value)
                    if not connected:
                        scan_button.configure(state="normal")
                        connect_button.configure(state="normal" if tree.get_children() else "disabled")
                elif kind == "data":
                    stats.set(f"累计 {value['bytes']} 字节   |   有效六列数据 {value['lines']} 行   |   "
                              f"当前 {value['byte_rate']} 字节/秒、{value['line_rate']} 行/秒   |   格式异常 {value['bad']} 行")
                    if value["latest"] is not None:
                        latest.set("最新数据：" + ", ".join(str(v) for v in value["latest"]))
                        status.set("正在接收传感器数据。此固件正常输出约 100 行/秒；这不是完整采样率。")
                    elif value["bytes"]:
                        status.set("已收到蓝牙字节，但尚未解析到六列数据；请核对烧录的固件和 BT24 串口波特率。")
                    elif value["seconds"] >= 3:
                        status.set("蓝牙已连接，但没有收到字节。请确认开发板已复位、PB14→BT24 RXD，且两端串口均为 115200。")
                    log.configure(state="normal")
                    log.delete("1.0", "end")
                    log.insert("end", "\n".join(value["preview"]))
                    log.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(100, poll)

    def close():
        nonlocal closing
        closing = True
        status.set("正在关闭蓝牙连接……")
        future = asyncio.run_coroutine_threadsafe(worker.shutdown(), worker.loop)
        deadline = time.monotonic() + 6

        def finish():
            if future.done() or time.monotonic() >= deadline:
                worker.loop.call_soon_threadsafe(worker.loop.stop)
                root.destroy()
            else:
                root.after(100, finish)

        finish()

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, poll)
    root.mainloop()


async def cli_scan():
    from bleak import BleakScanner
    found = await BleakScanner.discover(timeout=8, return_adv=True)
    print(f"BLE scan succeeded: {len(found)} devices found")
    candidates = [(d, a) for d, a in found.values() if is_candidate(d, a)]
    for device, advert in candidates:
        print(advert.local_name or device.name or "BT24", device.address, advert.rssi)
    if not candidates:
        print("BT24 not found. Turn off phone Bluetooth, power-cycle BT24, and scan again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", help="Scan only; do not open a window")
    args = parser.parse_args()
    if args.scan:
        asyncio.run(cli_scan())
    else:
        run_gui()
