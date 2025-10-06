"""Interactive Matplotlib helpers."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from PySide6.QtCore import Qt


class CtrlScrollZoom:
    """Add zoom and pan behaviour to Matplotlib canvases."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.axes_limits = {}
        self._drag_info: Optional[tuple] = None
        # Ensure the canvas can catch modifier-aware wheel events and key presses.
        try:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
        except AttributeError:
            pass
        self.cid_scroll = canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_key = canvas.mpl_connect("key_press_event", self.on_key_press)
        self.cid_press = canvas.mpl_connect("button_press_event", self.on_button_press)
        self.cid_release = canvas.mpl_connect("button_release_event", self.on_button_release)
        self.cid_motion = canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def register_axes(self, axes: Iterable, full_xlim: Optional[Sequence[float]] = None):
        if not axes:
            return
        if full_xlim is None:
            limits = tuple(float(v) for v in next(iter(axes)).get_xlim())
        else:
            limits = tuple(float(v) for v in full_xlim)
        self.axes_limits = {ax: limits for ax in axes}

    def on_scroll(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if self._modifier_active(event, Qt.ControlModifier):
            self._zoom(ax, event)

    def on_key_press(self, event):
        if not event.key:
            return
        key = event.key.lower()
        if key == "r":
            for ax, limits in self.axes_limits.items():
                ax.set_xlim(limits)
            self.canvas.draw_idle()

    def on_button_press(self, event):
        if event.inaxes is None or event.button != 1 or event.xdata is None:
            return
        ax = event.inaxes
        self._drag_info = (ax, float(event.xdata), tuple(float(v) for v in ax.get_xlim()))

    def on_mouse_move(self, event):
        if not self._drag_info or event.inaxes is None or event.xdata is None:
            return
        ax, start_x, (orig_left, orig_right) = self._drag_info
        if event.inaxes is not ax:
            return
        delta = float(event.xdata) - start_x
        left = orig_left - delta
        right = orig_right - delta
        limits = self.axes_limits.get(ax)
        if limits:
            data_left, data_right = limits
            span = right - left
            if span >= data_right - data_left:
                left, right = data_left, data_right
            else:
                if left < data_left:
                    left = data_left
                    right = data_left + span
                if right > data_right:
                    right = data_right
                    left = data_right - span
        ax.set_xlim(left, right)
        self.canvas.draw_idle()

    def on_button_release(self, event):
        if event.button != 1:
            return
        self._drag_info = None

    def _zoom(self, ax, event):
        if event.xdata is None:
            return
        base_scale = 0.8 if event.button == "up" else 1.25
        cur_left, cur_right = ax.get_xlim()
        cursor = float(event.xdata)
        left = cursor - (cursor - cur_left) * base_scale
        right = cursor + (cur_right - cursor) * base_scale
        limits = self.axes_limits.get(ax)
        if limits:
            data_left, data_right = limits
            span = max(data_right - data_left, 1e-9)
            min_span = span / 1000
        else:
            data_left, data_right = cur_left, cur_right
            min_span = (cur_right - cur_left) / 1000 if cur_right > cur_left else 1e-6
        if right - left < min_span:
            return
        if limits:
            width = right - left
            if width > (data_right - data_left):
                left, right = data_left, data_right
            else:
                if left < data_left:
                    left = data_left
                    right = data_left + width
                if right > data_right:
                    right = data_right
                    left = data_right - width
        ax.set_xlim(left, right)
        self.canvas.draw_idle()

    def _modifier_active(self, event, modifier):
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None:
            try:
                return bool(gui_event.modifiers() & modifier)
            except AttributeError:
                pass
        key = (event.key or "").lower()
        if modifier == Qt.ControlModifier:
            return "control" in key or "ctrl" in key
        return False
