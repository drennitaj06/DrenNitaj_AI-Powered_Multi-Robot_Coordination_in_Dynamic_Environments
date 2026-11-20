"""A small, dependency-free SVG plotting helper for the project overview figures.

Static, light-surface figures for reports. Every mark carries a ``<title>`` (native hover
tooltip). Colours follow a validated categorical order; text always uses text tokens.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

# ---------------------------------------------------------------- tokens (light surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
AXIS = "#b9b8b2"
# categorical slots 1-8 (validated order); the first three are validated for all pairs
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# neutral greys for non-learned comparators (identity carried by labels and marker shape)
NEUTRAL = ("#3d3c39", "#77766f", "#a8a7a0")
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 10_000:
        return f"{v:,.0f}".replace(",", " ")
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    if a >= 1:
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.2g}" if a < 10 else f"{v:.1f}"
    return f"{v:.3g}"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + step * 1e-6:
        out.append(round(v, 12))
        v += step
    return out


def log_ticks(lo: float, hi: float) -> list[float]:
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0**e for e in range(a, b + 1) if lo * 0.999 <= 10.0**e <= hi * 1.001]


def marker(cx: float, cy: float, color: str, *, shape: str = "circle", r: float = 4.5,
           title: str = "", hollow: bool = False) -> str:  # fmt: skip
    """One point marker; hollow markers keep the colour as a ring on the surface."""
    fill = SURFACE if hollow else color
    t = f"<title>{escape(title)}</title>" if title else ""
    stroke, sw = (color, 2) if hollow else (SURFACE, 1.5)
    style = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if shape == "square":
        s = r * 1.7
        return (f'<rect x="{cx - s / 2:.1f}" y="{cy - s / 2:.1f}" width="{s:.1f}" '
                f'height="{s:.1f}" rx="1" {style}>{t}</rect>')  # fmt: skip
    if shape == "diamond":
        s = r * 1.35
        pts = f"{cx:.1f},{cy - s:.1f} {cx + s:.1f},{cy:.1f} {cx:.1f},{cy + s:.1f} {cx - s:.1f},{cy:.1f}"
        return f'<polygon points="{pts}" {style}>{t}</polygon>'
    if shape == "triangle":
        s = r * 1.4
        pts = (f"{cx:.1f},{cy - s:.1f} {cx + s:.1f},{cy + s * 0.8:.1f} "
               f"{cx - s:.1f},{cy + s * 0.8:.1f}")  # fmt: skip
        return f'<polygon points="{pts}" {style}>{t}</polygon>'
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {style}>{t}</circle>'


@dataclass
class Axes:
    x0: float
    y0: float
    w: float
    h: float
    xlim: tuple[float, float] = (0.0, 1.0)
    ylim: tuple[float, float] = (0.0, 1.0)
    xlog: bool = False
    ylog: bool = False
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xcats: Sequence[str] | None = None  # categorical x axis (band scale)
    ycats: Sequence[str] | None = None  # categorical y axis (band scale)
    xticks: Sequence[float] | None = None
    yticks: Sequence[float] | None = None
    xticklabels: Sequence[str] | None = None
    parts: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ scales
    def sx(self, v: float) -> float:
        if self.xcats is not None:
            step = self.w / len(self.xcats)
            return self.x0 + step * (v + 0.5)
        lo, hi = self.xlim
        if self.xlog:
            lo, hi, v = math.log10(lo), math.log10(hi), math.log10(max(v, 1e-300))
        return self.x0 + (v - lo) / (hi - lo) * self.w

    def sy(self, v: float) -> float:
        if self.ycats is not None:
            step = self.h / len(self.ycats)
            return self.y0 + step * (v + 0.5)
        lo, hi = self.ylim
        if self.ylog:
            lo, hi, v = math.log10(lo), math.log10(hi), math.log10(max(v, 1e-300))
        return self.y0 + self.h - (v - lo) / (hi - lo) * self.h

    # ------------------------------------------------------------ marks
    def line(self, xs: Sequence[float], ys: Sequence[float], color: str, *, width: float = 2.0,
             dash: str | None = None, title: str = "") -> None:  # fmt: skip
        pts = " ".join(f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in zip(xs, ys, strict=True))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        t = f"<title>{escape(title)}</title>" if title else ""
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d}>{t}</polyline>'
        )

    def band(self, xs: Sequence[float], lo: Sequence[float], hi: Sequence[float], color: str,
             *, opacity: float = 0.16) -> None:  # fmt: skip
        up = [f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in zip(xs, hi, strict=True)]
        dn = [f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in zip(xs, lo, strict=True)]
        self.parts.append(
            f'<polygon points="{" ".join(up + dn[::-1])}" fill="{color}" '
            f'fill-opacity="{opacity}" stroke="none"/>'
        )

    def point(self, x: float, y: float, color: str, *, r: float = 4.5, shape: str = "circle",
              title: str = "", hollow: bool = False) -> None:  # fmt: skip
        self.parts.append(marker(self.sx(x), self.sy(y), color, shape=shape, r=r, title=title,
                                 hollow=hollow))  # fmt: skip

    def vrange(self, x: float, lo: float, hi: float, color: str, *, width: float = 2.0,
               title: str = "") -> None:  # fmt: skip
        t = f"<title>{escape(title)}</title>" if title else ""
        self.parts.append(
            f'<line x1="{self.sx(x):.1f}" x2="{self.sx(x):.1f}" y1="{self.sy(lo):.1f}" '
            f'y2="{self.sy(hi):.1f}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round">{t}</line>'
        )

    def hrange(self, y: float, lo: float, hi: float, color: str, *, width: float = 2.0,
               title: str = "") -> None:  # fmt: skip
        t = f"<title>{escape(title)}</title>" if title else ""
        self.parts.append(
            f'<line y1="{self.sy(y):.1f}" y2="{self.sy(y):.1f}" x1="{self.sx(lo):.1f}" '
            f'x2="{self.sx(hi):.1f}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round">{t}</line>'
        )

    def hbar(self, y: float, x_from: float, x_to: float, color: str, *, thick: float = 14,
             title: str = "") -> None:  # fmt: skip
        a, b = sorted((self.sx(x_from), self.sx(x_to)))
        t = f"<title>{escape(title)}</title>" if title else ""
        self.parts.append(
            f'<rect x="{a:.1f}" y="{self.sy(y) - thick / 2:.1f}" width="{max(b - a, 0.5):.1f}" '
            f'height="{thick}" fill="{color}" rx="2">{t}</rect>'
        )

    def vbar(self, x: float, y_from: float, y_to: float, color: str, *, thick: float = 14,
             title: str = "", offset: float = 0.0) -> None:  # fmt: skip
        a, b = sorted((self.sy(y_from), self.sy(y_to)))
        t = f"<title>{escape(title)}</title>" if title else ""
        self.parts.append(
            f'<rect x="{self.sx(x) + offset - thick / 2:.1f}" y="{a:.1f}" width="{thick}" '
            f'height="{max(b - a, 0.5):.1f}" fill="{color}" rx="2">{t}</rect>'
        )

    def hline(self, y: float, color: str = AXIS, *, dash: str | None = "4 3", width: float = 1.2,
              label: str = "") -> None:  # fmt: skip
        d = f' stroke-dasharray="{dash}"' if dash else ""
        yy = self.sy(y)
        self.parts.append(
            f'<line x1="{self.x0:.1f}" x2="{self.x0 + self.w:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>'
        )
        if label:
            self.text(self.x0 + self.w - 4, yy - 5, label, anchor="end", size=10.5, color=TEXT_2)

    def vline(self, x: float, color: str = AXIS, *, dash: str | None = "4 3", width: float = 1.2,
              label: str = "") -> None:  # fmt: skip
        d = f' stroke-dasharray="{dash}"' if dash else ""
        xx = self.sx(x)
        self.parts.append(
            f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{self.y0:.1f}" y2="{self.y0 + self.h:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>'
        )
        if label:
            self.text(xx + 4, self.y0 + self.h - 6, label, anchor="start", size=10.5, color=TEXT_2)

    def text(self, x: float, y: float, s: str, *, anchor: str = "start", size: float = 11,
             color: str = TEXT_2, weight: str = "normal") -> None:  # fmt: skip
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}">{escape(s)}</text>'
        )

    def label_at(self, x: float, y: float, s: str, *, dx: float = 7, dy: float = 4,
                 anchor: str = "start", size: float = 10.5, color: str = TEXT_2) -> None:  # fmt: skip
        self.text(self.sx(x) + dx, self.sy(y) + dy, s, anchor=anchor, size=size, color=color)

    # ------------------------------------------------------------ frame
    def bottom(self) -> float:
        """Lowest y used by the tick labels and the x-axis title."""
        extra = 12 * (max(len(str(c).split("\n")) for c in self.xcats) - 1) if self.xcats else 0
        return self.y0 + self.h + (40 if self.xlabel else 24) + extra

    def render(self) -> str:
        out: list[str] = []
        # y grid + ticks
        if self.ycats is not None:
            for i, c in enumerate(self.ycats):
                yy = self.sy(i)
                out.append(
                    f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'
                )
                out.append(
                    f'<text x="{self.x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="{TEXT}">{escape(c)}</text>'
                )
        else:
            yt = list(self.yticks) if self.yticks is not None else (
                log_ticks(*self.ylim) if self.ylog else nice_ticks(*self.ylim))  # fmt: skip
            for v in yt:
                yy = self.sy(v)
                out.append(
                    f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'
                )
                out.append(
                    f'<text x="{self.x0 - 7}" y="{yy + 3.8:.1f}" text-anchor="end" font-size="10.5" fill="{TEXT_2}">{fmt(v)}</text>'
                )
        # x ticks
        if self.xcats is not None:
            for i, c in enumerate(self.xcats):
                xx = self.sx(i)
                for k, part in enumerate(str(c).split("\n")):
                    out.append(
                        f'<text x="{xx:.1f}" y="{self.y0 + self.h + 16 + 12 * k:.1f}" text-anchor="middle" font-size="10.5" fill="{TEXT}">{escape(part)}</text>'
                    )
        else:
            xt = list(self.xticks) if self.xticks is not None else (
                log_ticks(*self.xlim) if self.xlog else nice_ticks(*self.xlim))  # fmt: skip
            labels = (
                list(self.xticklabels) if self.xticklabels is not None else [fmt(v) for v in xt]
            )
            for v, lab in zip(xt, labels, strict=True):
                xx = self.sx(v)
                out.append(
                    f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{self.y0}" y2="{self.y0 + self.h}" stroke="{GRID}" stroke-width="1"/>'
                )
                out.append(
                    f'<text x="{xx:.1f}" y="{self.y0 + self.h + 15:.1f}" text-anchor="middle" font-size="10.5" fill="{TEXT_2}">{escape(lab)}</text>'
                )
        # baseline axes
        out.append(
            f'<line x1="{self.x0}" x2="{self.x0 + self.w}" y1="{self.y0 + self.h}" y2="{self.y0 + self.h}" stroke="{AXIS}" stroke-width="1"/>'
        )
        out.append(
            f'<line x1="{self.x0}" x2="{self.x0}" y1="{self.y0}" y2="{self.y0 + self.h}" stroke="{AXIS}" stroke-width="1"/>'
        )
        if self.title:
            out.append(
                f'<text x="{self.x0}" y="{self.y0 - 12}" font-size="12.5" font-weight="600" fill="{TEXT}">{escape(self.title)}</text>'
            )
        if self.xlabel:
            extra = 12 * (max(len(str(c).split("\n")) for c in self.xcats) - 1) if self.xcats else 0
            out.append(
                f'<text x="{self.x0 + self.w / 2:.1f}" y="{self.y0 + self.h + 34 + extra:.1f}" text-anchor="middle" font-size="11" fill="{TEXT_2}">{escape(self.xlabel)}</text>'
            )
        if self.ylabel:
            cx, cy = self.x0 - 46, self.y0 + self.h / 2
            out.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="11" fill="{TEXT_2}" transform="rotate(-90 {cx:.1f} {cy:.1f})">{escape(self.ylabel)}</text>'
            )
        clip = f'clip-path="url(#c{id(self)})"'
        defs = f'<clipPath id="c{id(self)}"><rect x="{self.x0 - 8}" y="{self.y0 - 8}" width="{self.w + 16}" height="{self.h + 16}"/></clipPath>'
        return "\n".join([defs, *out, f"<g {clip}>", *self.parts, "</g>"])


@dataclass
class Figure:
    width: float
    height: float
    title: str
    subtitle: str = ""
    axes: list[Axes] = field(default_factory=list)
    legend: list[tuple[str, str, str]] = field(default_factory=list)  # (label, colour, shape)
    notes: list[str] = field(default_factory=list)

    def add(self, ax: Axes) -> Axes:
        self.axes.append(ax)
        return ax

    def svg(self) -> str:
        wrap = int((self.width - 48) / 5.7)
        sub = textwrap.wrap(self.subtitle, int((self.width - 48) / 6.2)) if self.subtitle else []
        notes = [ln for n in self.notes for ln in textwrap.wrap(n, wrap)]
        bottom = max(ax.bottom() for ax in self.axes)
        height = max(self.height, bottom + 18 + 15 * len(notes))
        body: list[str] = [
            f'<rect width="{self.width}" height="{height}" fill="{SURFACE}"/>',
            f'<text x="24" y="30" font-size="16" font-weight="650" fill="{TEXT}">{escape(self.title)}</text>',
        ]
        for k, line in enumerate(sub):
            body.append(
                f'<text x="24" y="{50 + 15 * k}" font-size="11.5" fill="{TEXT_2}">{escape(line)}</text>'
            )
        x = 24.0
        y = 72.0 + 15 * max(len(sub) - 1, 0) if sub else 54.0
        for label, color, shape in self.legend:
            if shape == "line":
                body.append(
                    f'<line x1="{x}" x2="{x + 18}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
                )
                x += 24
            elif shape == "dash":
                body.append(
                    f'<line x1="{x}" x2="{x + 18}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4 3"/>'
                )
                x += 24
            elif shape == "band":
                body.append(
                    f'<rect x="{x}" y="{y - 10}" width="18" height="11" fill="{color}" fill-opacity="0.25" rx="2"/>'
                )
                x += 24
            elif shape == "swatch":
                body.append(
                    f'<rect x="{x}" y="{y - 10}" width="11" height="11" fill="{color}" rx="2"/>'
                )
                x += 16
            else:
                hollow = shape.endswith("-hollow")
                body.append(marker(x + 6, y - 4, color, shape=shape.removesuffix("-hollow"), r=4.5,
                                   hollow=hollow))  # fmt: skip
                x += 16
            body.append(
                f'<text x="{x}" y="{y}" font-size="11.5" fill="{TEXT}">{escape(label)}</text>'
            )
            x += 7.2 * len(label) + 22
        for ax in self.axes:
            body.append(ax.render())
        ny = bottom + 18
        for n in notes:
            body.append(
                f'<text x="24" y="{ny:.1f}" font-size="10.5" fill="{MUTED}">{escape(n)}</text>'
            )
            ny += 15
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {height}" '
            f'width="{self.width}" height="{height}" font-family="{FONT}" role="img" '
            f'aria-label="{escape(self.title)}">\n<title>{escape(self.title)}</title>\n'
            + "\n".join(body)
            + "\n</svg>\n"
        )
