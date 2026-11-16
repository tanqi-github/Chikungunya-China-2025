"""Redraw the data-descriptor figures from the public city-week release only.

Core conclusion: reported chikungunya moved from sparse introductions to a
large local outbreak and then to a wider set of reporting cities, with each
week's cases concentrated in few cities.

This script does not read case-level files or the internal release tables.
Local counts, milestones and concentration are recomputed from the public
city-week file. Acceptance totals come from release_summary.json.

Public inputs, relative to --release-dir (default outputs/scidata/public_release):
  chikv_china2025_city_week.csv
  city_lookup.csv
  release_summary.json

Basemap: province and prefecture GeoJSON published by Tianditu
(天地图, 国家级地理信息公共服务平台, 行政区划). Download:
  https://cloudcenter.tianditu.gov.cn/administrativeDivision
Map approval number: GS（2024）0650号.
Required file names inside --basemap-dir:
  中国_省.geojson
  中国_市.geojson
The directory can also be set with SCIDATA_TIANDITU_DIR. The script does not
download the files. The on-map credit is "Base map: Tianditu, GS(2024)0650".

Archetypes: Fig. 1 and 5 are quantitative grids; Fig. 2 is a map plate;
Fig. 3 and 4 are asymmetric mixed figures.
Backend: Python / matplotlib only.
Export: SVG and PDF with editable text, plus a 600 dpi PNG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path as MplPath
from pypinyin import lazy_pinyin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = Path(
    os.environ.get("SCIDATA_PUBLIC_RELEASE", ROOT / "outputs" / "scidata" / "public_release")
)
DEFAULT_FIGURES = ROOT / "outputs" / "scidata" / "public_figures"
BASEMAP_DOWNLOAD = "https://cloudcenter.tianditu.gov.cn/administrativeDivision"
BASEMAP_FILES = ("中国_省.geojson", "中国_市.geojson")
ATTRIBUTION = "Base map: Tianditu, GS(2024)0650"
SEED = 20261026
FIGURES = DEFAULT_FIGURES
MUNICIPALITY_PREFIXES = {"11", "12", "31", "50"}
ADMIN_SUFFIXES = (
    "黎族苗族自治县",
    "苗族侗族自治州",
    "布依族苗族自治州",
    "壮族苗族自治州",
    "自治州",
    "自治县",
    "地区",
    "林区",
    "市",
    "县",
    "盟",
    "旗",
)
ETHNIC_SUFFIXES = (
    "黎族苗族",
    "苗族侗族",
    "布依族苗族",
    "壮族苗族",
    "柯尔克孜族",
    "维吾尔族",
    "哈萨克族",
    "蒙古族",
    "朝鲜族",
    "塔吉克族",
    "乌孜别克族",
    "达斡尔族",
    "鄂温克族",
    "俄罗斯族",
    "鄂伦春族",
    "布朗族",
    "傈僳族",
    "拉祜族",
    "景颇族",
    "东乡族",
    "撒拉族",
    "毛南族",
    "仡佬族",
    "阿昌族",
    "普米族",
    "怒族",
    "德昂族",
    "保安族",
    "裕固族",
    "独龙族",
    "赫哲族",
    "门巴族",
    "珞巴族",
    "基诺族",
    "锡伯族",
    "土族",
    "水族",
    "畲族",
    "佤族",
    "傣族",
    "哈尼族",
    "白族",
    "彝族",
    "藏族",
    "回族",
    "满族",
    "壮族",
    "苗族",
    "侗族",
    "黎族",
    "羌族",
    "京族",
    "瑶族",
)

MAIN_EXTENT = (72.6, 136.0, 17.55, 54.5)
INSET_EXTENT = (107.4, 123.6, 2.85, 25.3)
WEEK_MIN, WEEK_MAX = 1, 47

PUBLIC_SERIES = [
    ("inferred_local_cases", "Local", "#0072B2"),
    ("inferred_imported_cases", "Inferred imported", "#E69F00"),
]
PROVINCE_NAMES = {
    "11": "Beijing", "12": "Tianjin", "13": "Hebei", "14": "Shanxi",
    "15": "Inner Mongolia", "21": "Liaoning", "22": "Jilin",
    "23": "Heilongjiang", "31": "Shanghai", "32": "Jiangsu",
    "33": "Zhejiang", "34": "Anhui", "35": "Fujian", "36": "Jiangxi",
    "37": "Shandong", "41": "Henan", "42": "Hubei", "43": "Hunan",
    "44": "Guangdong", "45": "Guangxi", "46": "Hainan",
    "50": "Chongqing", "51": "Sichuan", "52": "Guizhou", "53": "Yunnan",
    "54": "Tibet", "61": "Shaanxi", "62": "Gansu", "63": "Qinghai",
    "64": "Ningxia", "65": "Xinjiang",
}
MILESTONE_COLORS = {"M0": "#4D4D4D", "L1": "#0072B2", "L5": "#E69F00", "L2w": "#CC79A7"}
MILESTONE_LABELS = {"M0": "First case", "L1": "First local", "L5": "Five local", "L2w": "Two local weeks"}
P_LEVELS = [30, 50, 70, 90, 100]
P_COLORS = {30: "#9ECAE1", 50: "#6BAED6", 70: "#4292C6", 90: "#2171B5", 100: "#084594"}
GD_COLOR = "#0072B2"
OTHER_COLOR = "#E69F00"
TIME_CMAP = plt.get_cmap("viridis")
CASE_CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))


@dataclass
class CityShape:
    path: MplPath
    bounds: tuple[float, float, float, float]
    name: str


@dataclass
class Basemap:
    provinces: list[MplPath]
    province_by_code: dict[str, MplPath]
    lines: list[np.ndarray]
    cities: dict[str, CityShape]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#272727",
            "text.color": "#272727",
            "axes.labelcolor": "#272727",
            "xtick.color": "#272727",
            "ytick.color": "#272727",
            "legend.frameon": False,
            "hatch.linewidth": 0.3,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def millimetres(width: float, height: float) -> tuple[float, float]:
    return width / 25.4, height / 25.4


def save_figure(fig: plt.Figure, stem: str, tight: bool = True) -> dict[str, int]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for suffix in (".svg", ".pdf", ".png"):
        path = FIGURES / f"{stem}{suffix}"
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        kwargs = {"facecolor": "white"}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": 0.04})
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(temporary, **kwargs)
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.2)
        sizes[suffix] = path.stat().st_size
    plt.close(fig)
    print(json.dumps({"figure": stem, "png_bytes": sizes[".png"], "pdf_bytes": sizes[".pdf"]}))
    return sizes


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AssertionError(f"missing {path.name}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def resolve_basemap_dir(path: Path | None) -> Path:
    if path is None:
        environment = os.environ.get("SCIDATA_TIANDITU_DIR", "").strip()
        path = Path(environment) if environment else None
    if path is None or not path.is_dir():
        raise SystemExit(
            "Tianditu province and city GeoJSON were not found.\n"
            "Download the administrative-division files from 天地图:\n"
            f"  {BASEMAP_DOWNLOAD}\n"
            "Map approval number: GS（2024）0650号.\n"
            "Place these files in one directory and pass it with --basemap-dir "
            "or SCIDATA_TIANDITU_DIR:\n"
            + "\n".join(f"  {name}" for name in BASEMAP_FILES)
        )
    missing = [name for name in BASEMAP_FILES if not (path / name).is_file()]
    if missing:
        raise SystemExit(
            "The basemap directory is missing "
            + ", ".join(missing)
            + ".\nDownload them from 天地图 administrative divisions:\n"
            f"  {BASEMAP_DOWNLOAD}\n"
            "Map approval number: GS（2024）0650号."
        )
    return path


def place_stem(name: str) -> str:
    stem = name.strip()
    changed = True
    suffixes = sorted(ADMIN_SUFFIXES + ETHNIC_SUFFIXES, key=len, reverse=True)
    while changed and stem:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break
    if "族" in stem:
        raise AssertionError(f"place name still contains an ethnic suffix: {stem}")
    return stem


def romanize_place(name: str) -> str:
    syllables = lazy_pinyin(place_stem(name))
    if not syllables:
        raise AssertionError("place name has no pinyin")
    parts: list[str] = []
    for index, syllable in enumerate(syllables):
        if index and syllable[:1].lower() in {"a", "e", "o"}:
            parts.append("'")
        parts.append(syllable)
    text = "".join(parts)
    return text[:1].upper() + text[1:]


def city_label(city_code: str, city_name_cn: str) -> str:
    province = PROVINCE_NAMES.get(city_code[:2])
    if province is None:
        raise AssertionError(f"missing province name for code prefix {city_code[:2]}")
    if city_code[:2] in MUNICIPALITY_PREFIXES:
        return province
    city = romanize_place(city_name_cn)
    if city.casefold() == province.casefold():
        return city
    return f"{city}, {province}"


def assert_label_rules() -> None:
    examples = {
        "西安市": "Xi'an",
        "六安市": "Lu'an",
        "吉安市": "Ji'an",
        "广安市": "Guang'an",
        "普洱市": "Pu'er",
        "黔东南苗族侗族自治州": "Qiandongnan",
        "黔南布依族苗族自治州": "Qiannan",
        "文山壮族苗族自治州": "Wenshan",
        "昌江黎族自治县": "Changjiang",
        "保亭黎族苗族自治县": "Baoting",
    }
    for name, expected in examples.items():
        if romanize_place(name) != expected:
            raise AssertionError(f"romanization of {name} is not {expected}")
    if city_label("110000", "北京市") != "Beijing":
        raise AssertionError("municipality label repeated the city name")
    if city_label("500000", "重庆市") != "Chongqing":
        raise AssertionError("Chongqing label repeated the city name")
    if city_label("440100", "广州市") != "Guangzhou, Guangdong":
        raise AssertionError("prefecture label did not keep the province")
    if city_label("350100", "福州市") != "Fuzhou, Fujian":
        raise AssertionError("Fuzhou, Fujian label changed")
    if city_label("361000", "抚州市") != "Fuzhou, Jiangxi":
        raise AssertionError("Fuzhou, Jiangxi label changed")


def week_table(week_ids: pd.Series) -> pd.DataFrame:
    unique = sorted(set(week_ids))
    rows = []
    for week_id in unique:
        year_text, week_text = week_id.split("-W")
        start = pd.Timestamp.fromisocalendar(int(year_text), int(week_text), 1)
        rows.append({"week_id": week_id, "week_start": start})
    weeks = pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
    gaps = weeks["week_start"].diff().dt.days.iloc[1:]
    if not (gaps == 7).all():
        raise AssertionError("public week identifiers are not consecutive ISO weeks")
    weeks["week_index"] = np.arange(1, len(weeks) + 1)
    weeks["week_start"] = weeks["week_start"].dt.strftime("%Y-%m-%d")
    return weeks


def first_positive(counts: np.ndarray) -> int | None:
    hits = np.flatnonzero(counts > 0)
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def cumulative_at_least(counts: np.ndarray, threshold: int) -> int | None:
    hits = np.flatnonzero(np.cumsum(counts) >= threshold)
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def consecutive_positive(counts: np.ndarray) -> int | None:
    paired = (counts[:-1] > 0) & (counts[1:] > 0)
    hits = np.flatnonzero(paired)
    if len(hits) == 0:
        return None
    return int(hits[0]) + 1


def milestone_value(index: int | None) -> str:
    return "" if index is None else str(index)


def build_milestones(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    week_columns = list(range(1, int(panel["week_index"].max()) + 1))
    for city_code, city in panel.groupby("city_code", sort=True):
        ordered = city.set_index("week_index").reindex(week_columns)
        if ordered[["reported_cases", "inferred_local_cases"]].isna().any().any():
            raise AssertionError("a reporting city is missing a week")
        reported = ordered["reported_cases"].to_numpy(dtype=float)
        local = ordered["inferred_local_cases"].to_numpy(dtype=float)
        code = str(city_code)
        records.append(
            {
                "city_code": code,
                "is_guangdong": int(code.startswith("44")),
                "M0_week_index": milestone_value(first_positive(reported)),
                "L1_week_index": milestone_value(first_positive(local)),
                "L5_week_index": milestone_value(cumulative_at_least(local, 5)),
                "L2w_week_index": milestone_value(consecutive_positive(local)),
            }
        )
    milestones = pd.DataFrame(records)
    if milestones["M0_week_index"].eq("").any():
        raise AssertionError("a reporting city has no reported case")
    return milestones


def load_tables(release_dir: Path) -> dict[str, pd.DataFrame]:
    summary_path = release_dir / "release_summary.json"
    if not summary_path.is_file():
        raise AssertionError(f"missing {summary_path.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("assertions_passed"):
        raise AssertionError("release_summary.json did not pass its checks")
    public_panel = read_csv(release_dir / "chikv_china2025_city_week.csv")
    lookup = read_csv(release_dir / "city_lookup.csv")
    required = {"city_code", "week_id", "reported_cases", "inferred_imported_cases"}
    missing = sorted(required - set(public_panel.columns))
    if missing:
        raise AssertionError(f"public city-week table is missing {missing}")
    if list(lookup.columns) != ["city_code", "city_name_cn"]:
        raise AssertionError("city_lookup.csv columns are not city_code, city_name_cn")
    for column in ("reported_cases", "inferred_imported_cases"):
        public_panel[column] = pd.to_numeric(public_panel[column], errors="raise").astype("int64")
    if (public_panel["reported_cases"] < 0).any() or (public_panel["inferred_imported_cases"] < 0).any():
        raise AssertionError("public case counts contain a negative value")
    public_panel["inferred_local_cases"] = (
        public_panel["reported_cases"] - public_panel["inferred_imported_cases"]
    )
    if (public_panel["inferred_local_cases"] < 0).any():
        raise AssertionError("inferred imported cases cannot exceed reported cases")
    weeks = week_table(public_panel["week_id"])
    public_panel = public_panel.merge(
        weeks[["week_id", "week_index"]], on="week_id", how="left", validate="many_to_one"
    )
    expected = {
        "rows": len(public_panel),
        "cities": int(public_panel["city_code"].nunique()),
        "weeks": int(public_panel["week_id"].nunique()),
        "reported_cases": int(public_panel["reported_cases"].sum()),
        "inferred_imported_cases": int(public_panel["inferred_imported_cases"].sum()),
        "city_lookup_rows": len(lookup),
    }
    for key, value in expected.items():
        if int(summary[key]) != value:
            raise AssertionError(f"release_summary.json {key} does not match the public tables")
    if len(public_panel) != expected["cities"] * expected["weeks"]:
        raise AssertionError("public city-week table is not a complete city-by-week grid")
    if not lookup["city_code"].is_unique or set(lookup["city_code"]) != set(public_panel["city_code"]):
        raise AssertionError("city_lookup.csv does not match the reporting-city codes")
    labels = {
        row.city_code: city_label(row.city_code, row.city_name_cn)
        for row in lookup.itertuples(index=False)
    }
    panel = public_panel.loc[:, ["city_code", "week_index", "reported_cases"]].rename(
        columns={"reported_cases": "cases_total"}
    )
    milestones = build_milestones(public_panel)
    return {
        "weeks": weeks.sort_values("week_index"),
        "panel": panel,
        "public_panel": public_panel,
        "milestones": milestones,
        "labels": labels,
        "summary": summary,
    }


def gb_city_code(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if digits.startswith("156"):
        digits = digits[3:]
    return digits[:6]


def iter_rings(geometry: dict):
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "Polygon":
        yield from coordinates
    elif kind == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def iter_lines(geometry: dict):
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "LineString":
        yield coordinates
    elif kind == "MultiLineString":
        yield from coordinates


def path_from_geometry(geometry: dict) -> tuple[MplPath, tuple[float, float, float, float]] | None:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    longitudes: list[float] = []
    latitudes: list[float] = []
    for ring in iter_rings(geometry):
        points = [(float(point[0]), float(point[1])) for point in ring]
        if len(points) >= 2 and points[0] == points[-1]:
            points = points[:-1]
        if len(points) < 3:
            continue
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(points) - 1))
        codes.append(MplPath.CLOSEPOLY)
        vertices.extend(points)
        vertices.append(points[0])
        longitudes.extend(point[0] for point in points)
        latitudes.extend(point[1] for point in points)
    if not vertices:
        return None
    bounds = (min(longitudes), min(latitudes), max(longitudes), max(latitudes))
    return MplPath(vertices, codes), bounds


def point_in_extent(longitude: float, latitude: float, extent: tuple[float, float, float, float], margin: float = 0.08) -> bool:
    return extent[0] - margin <= longitude <= extent[1] + margin and extent[2] - margin <= latitude <= extent[3] + margin


def assert_coverage(province_path: Path) -> None:
    payload = json.loads(province_path.read_text(encoding="utf-8"))
    boundary_vertices = 0
    for feature in payload["features"]:
        geometry = feature.get("geometry") or {}
        name = str((feature.get("properties") or {}).get("name", ""))
        kind = geometry.get("type")
        if kind in {"Polygon", "MultiPolygon"}:
            for ring in iter_rings(geometry):
                latitudes = [float(point[1]) for point in ring]
                longitudes = [float(point[0]) for point in ring]
                minimum = min(latitudes)
                if minimum >= 18:
                    target = MAIN_EXTENT
                elif minimum < 17.5:
                    target = INSET_EXTENT
                else:
                    raise AssertionError("a province ring falls between the main map and the inset")
                if not all(point_in_extent(lon, lat, target) for lon, lat in zip(longitudes, latitudes)):
                    raise AssertionError(
                        "province ring outside its map extent: "
                        f"{min(longitudes):.3f},{minimum:.3f},{max(longitudes):.3f},{max(latitudes):.3f}"
                    )
        if name == "境界线":
            for line in iter_lines(geometry):
                for point in line:
                    longitude, latitude = float(point[0]), float(point[1])
                    boundary_vertices += 1
                    target = INSET_EXTENT if latitude < 17.55 else MAIN_EXTENT
                    if not point_in_extent(longitude, latitude, target):
                        raise AssertionError(f"boundary vertex outside its map extent: {longitude:.3f},{latitude:.3f}")
    if boundary_vertices == 0:
        raise AssertionError("national boundary lines were not found")


def load_basemap(province_path: Path, city_path: Path) -> Basemap:
    assert_coverage(province_path)
    provinces_payload = json.loads(province_path.read_text(encoding="utf-8"))
    province_paths: list[MplPath] = []
    province_by_code: dict[str, MplPath] = {}
    lines: list[np.ndarray] = []
    for feature in provinces_payload["features"]:
        geometry = feature.get("geometry") or {}
        name = str((feature.get("properties") or {}).get("name", ""))
        if name == "境界线" or geometry.get("type") in {"LineString", "MultiLineString"}:
            for line in iter_lines(geometry):
                array = np.asarray([(float(point[0]), float(point[1])) for point in line], dtype=float)
                if len(array) >= 2:
                    lines.append(array)
            continue
        parsed = path_from_geometry(geometry)
        if parsed is not None:
            province_paths.append(parsed[0])
            code = gb_city_code((feature.get("properties") or {}).get("gb", ""))
            if len(code) == 6:
                province_by_code[code] = parsed[0]
    cities_payload = json.loads(city_path.read_text(encoding="utf-8"))
    cities: dict[str, CityShape] = {}
    for feature in cities_payload["features"]:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        code = gb_city_code((feature.get("properties") or {}).get("gb", ""))
        parsed = path_from_geometry(geometry)
        if parsed is None or len(code) != 6:
            continue
        if code in cities:
            raise AssertionError("duplicate city polygon")
        name = str((feature.get("properties") or {}).get("name", "")).strip()
        cities[code] = CityShape(parsed[0], parsed[1], name)
    if not province_paths or not lines:
        raise AssertionError("basemap polygons or boundary lines are empty")
    return Basemap(province_paths, province_by_code, lines, cities)


def assert_cities_match(basemap: Basemap, milestones: pd.DataFrame) -> None:
    codes = set(milestones["city_code"])
    missing = codes - set(basemap.cities)
    if missing or len(codes) != 157:
        raise AssertionError(f"basemap match {len(codes) - len(missing)} of {len(codes)}")
    outside = 0
    for code in codes:
        bounds = basemap.cities[code].bounds
        corners = (
            (bounds[0], bounds[1]),
            (bounds[0], bounds[3]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
        )
        if not all(point_in_extent(lon, lat, MAIN_EXTENT) for lon, lat in corners):
            outside += 1
    if outside:
        raise AssertionError(f"{outside} reporting-city polygons fall outside the main map")


def month_ticks(weeks: pd.DataFrame, step: int = 1) -> tuple[list[int], list[str]]:
    ticks: list[int] = []
    labels: list[str] = []
    seen: set[int] = set()
    for row in weeks.itertuples(index=False):
        stamp = pd.Timestamp(row.week_start)
        if stamp.month in seen:
            continue
        seen.add(stamp.month)
        ticks.append(int(row.week_index))
        labels.append(stamp.strftime("%b"))
    return ticks[::step], labels[::step]


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.02,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


def finish_map(ax: plt.Axes, extent: tuple[float, float, float, float], frame: bool, attribution: bool) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    midpoint = (extent[2] + extent[3]) / 2
    ax.set_aspect(1 / math.cos(math.radians(midpoint)), adjustable="box")
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(frame)
        if frame:
            spine.set_linewidth(0.4)
            spine.set_color("#4D4D4D")
    ax.set_axisbelow(False)
    if attribution:
        ax.text(
            0.012,
            0.012,
            ATTRIBUTION,
            transform=ax.transAxes,
            fontsize=5,
            color="#4D4D4D",
            ha="left",
            va="bottom",
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
        )
    ax.autoscale(False)


def render_map(
    ax: plt.Axes,
    basemap: Basemap,
    extent: tuple[float, float, float, float],
    week_by_city: dict[str, float | None],
    frame: bool,
    attribution: bool,
    norm: Normalize | None = None,
) -> None:
    for path in basemap.provinces:
        ax.add_patch(PathPatch(path, facecolor="#FBFBFA", edgecolor="#D9D9D9", linewidth=0.25, zorder=1))
    cmap = TIME_CMAP
    if norm is None:
        norm = Normalize(vmin=WEEK_MIN, vmax=WEEK_MAX)
    for code, week_index in week_by_city.items():
        city = basemap.cities[code]
        if week_index is None:
            face, hatch, edge = "#E6E6E6", "///", "#8A8A8A"
        else:
            face, hatch, edge = cmap(norm(week_index)), None, "#6A6A6A"
        ax.add_patch(
            PathPatch(
                city.path,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.12,
                hatch=hatch,
                zorder=3,
            )
        )
    if basemap.lines:
        ax.add_collection(LineCollection(basemap.lines, colors="#6E6E6E", linewidths=0.4, zorder=4))
    finish_map(ax, extent, frame=frame, attribution=attribution)


def render_count_map(
    ax: plt.Axes,
    basemap: Basemap,
    extent: tuple[float, float, float, float],
    count_by_city: dict[str, int],
    norm: LogNorm,
    frame: bool,
    attribution: bool,
) -> None:
    for path in basemap.provinces:
        ax.add_patch(PathPatch(path, facecolor="#FBFBFA", edgecolor="#D9D9D9", linewidth=0.25, zorder=1))
    for code, count in count_by_city.items():
        city = basemap.cities[code]
        ax.add_patch(
            PathPatch(
                city.path,
                facecolor=CASE_CMAP(norm(count)),
                edgecolor="#6A6A6A",
                linewidth=0.12,
                zorder=3,
            )
        )
    if basemap.lines:
        ax.add_collection(LineCollection(basemap.lines, colors="#6E6E6E", linewidths=0.4, zorder=4))
    finish_map(ax, extent, frame=frame, attribution=attribution)


def emphasize_province_boundary(
    ax: plt.Axes,
    basemap: Basemap,
    code: str,
    label: str,
) -> None:
    path = basemap.province_by_code.get(code)
    if path is None:
        raise AssertionError(f"province boundary is missing for {code}")
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor="#FFFFFF",
            linewidth=1.65,
            zorder=5,
        )
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor="#3F3F3F",
            linewidth=0.80,
            zorder=6,
        )
    )
    vertices = path.vertices[np.isfinite(path.vertices).all(axis=1)]
    left, bottom = vertices.min(axis=0)
    right, top = vertices.max(axis=0)
    centre_y = (bottom + top) / 2
    ax.annotate(
        label,
        xy=(right, centre_y),
        xytext=(ax.get_xlim()[1] - 1.0, centre_y + 0.2),
        ha="right",
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color="#3F3F3F",
        arrowprops={"arrowstyle": "-", "color": "#3F3F3F", "linewidth": 0.55},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.5},
        zorder=7,
    )


def guangdong_extent(basemap: Basemap, codes: list[str]) -> tuple[float, float, float, float]:
    bounds = [basemap.cities[code].bounds for code in codes]
    margin = 0.4
    return (
        min(item[0] for item in bounds) - margin,
        max(item[2] for item in bounds) + margin,
        min(item[1] for item in bounds) - margin,
        max(item[3] for item in bounds) + margin,
    )


def week_lookup(milestones: pd.DataFrame, column: str, codes: list[str]) -> dict[str, float | None]:
    indexed = milestones.set_index("city_code")
    lookup: dict[str, float | None] = {}
    for code in codes:
        raw = str(indexed.at[code, column])
        lookup[code] = None if raw == "" else float(raw)
    return lookup


def add_national_panel(fig: plt.Figure, cell) -> tuple[plt.Axes, plt.Axes]:
    inner = cell.subgridspec(1, 2, width_ratios=[5.6, 1.15], wspace=0.045)
    main = fig.add_subplot(inner[0, 0])
    rail = inner[0, 1].subgridspec(3, 1, height_ratios=[1.35, 0.06, 1.45], hspace=0.0)
    inset = fig.add_subplot(rail[2, 0])
    return main, inset


def style_time_axis(ax: plt.Axes, weeks: pd.DataFrame, step: int = 1) -> None:
    ticks, labels = month_ticks(weeks, step=step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    last = int(weeks["week_index"].max())
    ax.set_xlim(0.4, last + 0.6)
    ax.set_xlabel("Analysis week, 2025")


def milestone_scale(
    milestones: pd.DataFrame,
    columns: list[str],
    weeks: pd.DataFrame,
) -> tuple[Normalize, list[float], list[str]]:
    values = pd.concat([pd.to_numeric(milestones[column], errors="coerce") for column in columns]).dropna()
    vmin = float(values.min())
    vmax = float(values.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    indexed_weeks = weeks.set_index("week_index")

    def label_for(value: float) -> str:
        nearest = indexed_weeks.index[np.abs(indexed_weeks.index.to_numpy(dtype=float) - value).argmin()]
        return pd.to_datetime(indexed_weeks.at[nearest, "week_start"]).strftime("%b")

    base_ticks, base_labels = month_ticks(weeks, step=2)
    pairs: list[tuple[float, str]] = [(vmin, label_for(vmin))]
    pairs.extend((float(tick), label) for tick, label in zip(base_ticks, base_labels) if vmin < tick < vmax)
    end_label = label_for(vmax)
    if end_label != pairs[-1][1]:
        pairs.append((vmax, end_label))
    return norm, [item[0] for item in pairs], [item[1] for item in pairs]


def add_week_colorbar(
    fig: plt.Figure,
    axes: list[plt.Axes],
    norm: Normalize | None = None,
    ticks: list[float] | None = None,
    labels: list[str] | None = None,
    cax: plt.Axes | None = None,
    show_note: bool = True,
) -> None:
    if norm is None:
        norm = Normalize(vmin=WEEK_MIN, vmax=WEEK_MAX)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=TIME_CMAP)
    mappable.set_array([])
    if cax is None:
        colorbar = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.015)
    else:
        colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label("Milestone week", fontsize=6)
    if ticks is None or labels is None:
        raise AssertionError("week color bar requires ticks derived from the public week index")
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels)
    colorbar.ax.tick_params(labelsize=5, length=2, width=0.4)
    if show_note:
        colorbar.ax.text(
            0.5,
            -0.04,
            "White: no included record\nGrey hatch: included city,\nmilestone not reached",
            transform=colorbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=5,
            color="#4D4D4D",
            clip_on=False,
        )


def map_title(ax: plt.Axes, letter: str, title: str, reached: int, denominator: int) -> None:
    panel_letter(ax, letter)
    ax.text(0.0, 1.045, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, clip_on=False)
    ax.text(
        1.0,
        1.045,
        f"n = {reached} / {denominator}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        clip_on=False,
    )


def draw_milestone_map(
    fig: plt.Figure,
    cell,
    basemap: Basemap,
    milestones: pd.DataFrame,
    column: str,
    letter: str,
    title: str,
    guangdong_only: bool,
    norm: Normalize | None = None,
) -> plt.Axes:
    codes = milestones.loc[milestones["city_code"].str.startswith("44"), "city_code"].tolist() if guangdong_only else milestones["city_code"].tolist()
    if guangdong_only:
        ax = fig.add_subplot(cell)
        extent = guangdong_extent(basemap, codes)
        inset = None
    else:
        ax, inset = add_national_panel(fig, cell)
        extent = MAIN_EXTENT
    lookup = week_lookup(milestones, column, codes)
    render_map(ax, basemap, extent, lookup, frame=False, attribution=True, norm=norm)
    if inset is not None:
        render_map(inset, basemap, INSET_EXTENT, {}, frame=True, attribution=False, norm=norm)
    reached = sum(value is not None for value in lookup.values())
    denominator = len(codes)
    map_title(ax, letter, title, reached, denominator)
    return ax


def draw_top_city_profiles(
    fig: plt.Figure,
    cell,
    tables: dict[str, pd.DataFrame],
    basemap: Basemap,
    letter: str,
) -> list[plt.Axes]:
    weeks = tables["weeks"]
    public_panel = tables["public_panel"].copy()
    totals = public_panel.groupby("city_code")["reported_cases"].sum().sort_values(ascending=False)
    top_codes = totals.head(10).index.tolist()
    curves = cell.subgridspec(5, 2, hspace=0.28, wspace=0.16)
    ticks, labels = month_ticks(weeks, step=2)
    curve_axes: list[plt.Axes] = []
    for rank, code in enumerate(top_codes):
        ax_city = fig.add_subplot(curves[rank // 2, rank % 2])
        last_week = int(weeks["week_index"].max())
        week_axis = list(range(1, last_week + 1))
        city = (
            public_panel.loc[public_panel["city_code"].eq(code)]
            .set_index("week_index")
            .reindex(week_axis, fill_value=0)
        )
        x_city = np.arange(1, last_week + 1)
        bottom = np.zeros(last_week)
        city_series = [
            ("inferred_local_cases", "Local", GD_COLOR),
            ("inferred_imported_cases", "Inferred imported", OTHER_COLOR),
        ]
        for column, label, color in city_series:
            height = city[column].to_numpy(dtype=float)
            ax_city.bar(x_city, height, bottom=bottom, width=0.92, color=color, linewidth=0, label=label)
            bottom += height
        ax_city.text(
            0.035,
            0.94,
            f"{rank + 1}. {tables['labels'][code]}",
            transform=ax_city.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
            fontweight="bold",
        )
        ax_city.text(
            1.0,
            0.94,
            f"n = {int(totals.loc[code]):,}",
            transform=ax_city.transAxes,
            ha="right",
            va="top",
            fontsize=5.3,
            color="#4D4D4D",
        )
        ax_city.set_xlim(0.5, last_week + 0.5)
        ax_city.set_ylim(0, max(1, float(bottom.max()) * 1.20))
        ax_city.set_yticks([0, int(math.ceil(bottom.max()))])
        ax_city.tick_params(axis="y", labelsize=4.8, length=2, pad=1)
        if rank // 2 == 4:
            ax_city.set_xticks(ticks)
            ax_city.set_xticklabels(labels, fontsize=4.8)
        else:
            ax_city.set_xticks([])
        if rank % 2 == 0:
            ax_city.set_ylabel("Weekly cases", fontsize=5.2)
        ax_city.spines["left"].set_linewidth(0.5)
        ax_city.spines["bottom"].set_linewidth(0.5)
        curve_axes.append(ax_city)
    curve_axes[0].legend(loc="upper center", bbox_to_anchor=(1.04, 1.42), ncol=2, fontsize=5.3, frameon=False)
    panel_letter(curve_axes[0], letter)
    return curve_axes


def figure_1(tables: dict[str, pd.DataFrame], basemap: Basemap) -> None:
    weeks = tables["weeks"]
    panel = tables["public_panel"].copy()
    weekly = panel.groupby("week_index", as_index=False)[[column for column, _, _ in PUBLIC_SERIES]].sum()
    weekly = weeks[["week_index"]].merge(weekly, on="week_index", how="left").fillna(0)
    cities = (
        panel.loc[panel["reported_cases"] > 0]
        .groupby("week_index")["city_code"]
        .nunique()
        .reindex(weeks["week_index"], fill_value=0)
    )
    prefix = panel.assign(prefix44=panel["city_code"].str.startswith("44"))
    grouped = prefix.groupby(["week_index", "prefix44"])["inferred_local_cases"].sum().unstack(fill_value=0)
    fig = plt.figure(figsize=millimetres(183, 210))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.40, 2.55, 0.82],
        hspace=0.42,
        left=0.08,
        right=0.94,
        top=0.97,
        bottom=0.05,
    )
    ax = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[2], sharex=ax)
    x = weekly["week_index"].to_numpy()
    bottom = np.zeros(len(weekly))
    for column, label, color in PUBLIC_SERIES:
        height = weekly[column].to_numpy(dtype=float)
        ax.bar(x, height, bottom=bottom, width=0.86, color=color, label=label, linewidth=0)
        bottom += height
    ax.set_ylabel("Reported cases")
    ax.set_ylim(0, bottom.max() * 1.08)
    line = ax.twinx()
    line.plot(x, cities.to_numpy(), color="#4D4D4D", lw=1.0, label="Cities with ≥1 case")
    line.set_ylabel("Reporting cities", color="#4D4D4D")
    line.tick_params(axis="y", colors="#4D4D4D", labelsize=6)
    line.spines["right"].set_visible(True)
    line.spines["right"].set_color("#4D4D4D")
    line.spines["top"].set_visible(False)
    line.set_ylim(0, max(int(panel["city_code"].nunique()), int(cities.max()) * 1.15))
    handles, labels = ax.get_legend_handles_labels()
    extra, extra_labels = line.get_legend_handles_labels()
    ax.legend(handles + extra, labels + extra_labels, loc="upper left", ncol=2, fontsize=6)
    panel_letter(ax, "a")
    draw_top_city_profiles(fig, grid[1], tables, basemap, "b")
    for column, label, color in (
        (True, "Guangdong Province", GD_COLOR),
        (False, "Other provinces", OTHER_COLOR),
    ):
        if column not in grouped.columns:
            continue
        series = grouped[column].reindex(weeks["week_index"], fill_value=0).astype(float)
        ax2.plot(
            weeks["week_index"],
            series,
            color=color,
            lw=1.0,
            marker="o",
            ms=2.2,
            label=label,
        )
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Inferred local cases")
    ax2.legend(loc="upper left", fontsize=6)
    panel_letter(ax2, "c")
    style_time_axis(ax2, weeks)
    style_time_axis(ax, weeks)
    save_figure(fig, "fig1_weekly_curve")


def figure_maps(tables: dict[str, pd.DataFrame], basemap: Basemap, spec: list[tuple[str, str, str, bool]], stem: str, size: tuple[float, float], rows: int, cols: int) -> None:
    fig = plt.figure(figsize=millimetres(*size))
    grid = fig.add_gridspec(rows, cols, wspace=0.12, hspace=0.18, left=0.03, right=0.90, top=0.94, bottom=0.03)
    axes = []
    for position, (column, letter, title, guangdong_only) in enumerate(spec):
        cell = grid[position // cols, position % cols]
        axes.append(draw_milestone_map(fig, cell, basemap, tables["milestones"], column, letter, title, guangdong_only))
    add_week_colorbar(fig, axes)
    save_figure(fig, stem)


def figure_2(tables: dict[str, pd.DataFrame], basemap: Basemap) -> None:
    panel = tables["panel"]
    milestones = tables["milestones"]
    city_counts = panel.groupby("city_code")["cases_total"].sum().astype(int)
    count_lookup = city_counts.to_dict()
    count_norm = LogNorm(vmin=1, vmax=int(city_counts.max()))
    time_norm, time_ticks, time_labels = milestone_scale(
        milestones,
        ["L1_week_index", "L5_week_index", "L2w_week_index"],
        tables["weeks"],
    )

    fig = plt.figure(figsize=millimetres(183, 170))
    grid = fig.add_gridspec(2, 2, wspace=0.14, hspace=0.20, left=0.03, right=0.88, top=0.95, bottom=0.07)

    ax_count, inset_count = add_national_panel(fig, grid[0, 0])
    render_count_map(ax_count, basemap, MAIN_EXTENT, count_lookup, count_norm, frame=False, attribution=True)
    render_count_map(inset_count, basemap, INSET_EXTENT, {}, count_norm, frame=True, attribution=False)
    emphasize_province_boundary(ax_count, basemap, "440000", "Guangdong Province")
    panel_letter(ax_count, "a")
    ax_count.text(0.0, 1.045, "Cumulative reported cases", transform=ax_count.transAxes, ha="left", va="bottom", fontsize=7, clip_on=False)
    ax_count.text(1.0, 1.045, f"n = {int(city_counts.sum()):,}", transform=ax_count.transAxes, ha="right", va="bottom", fontsize=6, clip_on=False)

    milestone_specs = [
        ("L1_week_index", "b", "First local case", grid[0, 1]),
        ("L5_week_index", "c", "Five cumulative local cases", grid[1, 0]),
        ("L2w_week_index", "d", "Two consecutive local weeks", grid[1, 1]),
    ]
    time_axes = []
    for column, letter, title, cell in milestone_specs:
        time_axes.append(
            draw_milestone_map(
                fig,
                cell,
                basemap,
                milestones,
                column,
                letter,
                title,
                False,
                time_norm,
            )
        )

    count_mappable = plt.cm.ScalarMappable(norm=count_norm, cmap=CASE_CMAP)
    count_mappable.set_array([])
    count_cax = ax_count.inset_axes([0.0, -0.105, 1.0, 0.035])
    count_bar = fig.colorbar(count_mappable, cax=count_cax, orientation="horizontal")
    count_ticks = [value for value in (1, 10, 100, 1000, 10000) if value <= city_counts.max()]
    count_bar.set_ticks(count_ticks)
    count_bar.set_ticklabels([f"{value:,}" for value in count_ticks])
    count_bar.set_label("Cumulative cases per reporting city (log scale)", fontsize=6)
    count_bar.ax.tick_params(labelsize=5, length=2, width=0.4)
    count_bar.outline.set_linewidth(0.4)

    colorbar_ax = fig.add_axes([0.905, 0.22, 0.014, 0.60])
    add_week_colorbar(
        fig,
        time_axes,
        norm=time_norm,
        ticks=time_ticks,
        labels=time_labels,
        cax=colorbar_ax,
        show_note=False,
    )

    save_figure(fig, "fig2_first_timing")


def figure_4(tables: dict[str, pd.DataFrame]) -> None:
    weeks = tables["weeks"]
    milestones = tables["milestones"].copy()
    milestones["prefix44"] = milestones["is_guangdong"].eq(1)
    fig = plt.figure(figsize=millimetres(183, 128))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 0.95],
        hspace=0.48,
        left=0.10,
        right=0.96,
        top=0.95,
        bottom=0.12,
    )
    top = grid[0].subgridspec(1, 2, wspace=0.28)
    axes_a = []
    for position, (prefix, title) in enumerate(((True, "Guangdong Province"), (False, "Other provinces"))):
        ax = fig.add_subplot(top[0, position])
        subset = milestones.loc[milestones["prefix44"].eq(prefix)]
        last_week = int(weeks["week_index"].max())
        for code in ("M0", "L1", "L5", "L2w"):
            index = pd.to_numeric(subset[f"{code}_week_index"], errors="coerce")
            cumulative = [int((index <= week).sum()) for week in range(1, last_week + 1)]
            extended_x = list(range(1, last_week + 1)) + [last_week + 1]
            extended_y = cumulative + [cumulative[-1]]
            ax.plot(
                extended_x,
                extended_y,
                drawstyle="steps-post",
                color=MILESTONE_COLORS[code],
                lw=1.05,
                label=MILESTONE_LABELS[code],
            )
        ax.set_xlim(0.5, last_week + 0.8)
        ax.set_ylim(0, max(1, len(subset)) * 1.08)
        ax.set_ylabel("Cumulative cities")
        ax.set_title(f"{title} (n = {len(subset)})", loc="right" if position == 1 else "left", fontsize=7)
        style_time_axis(ax, weeks, step=2)
        if position == 0:
            panel_letter(ax, "a")
        if position == 1:
            ax.legend(loc="upper left", fontsize=5.5, borderpad=0.2, labelspacing=0.25)
        axes_a.append(ax)

    ax_c = fig.add_subplot(grid[1])
    rng = np.random.default_rng(SEED)
    labels_c = []
    sample_sizes: list[int] = []
    cursor = 0
    for code, label in (("L1", "To first local"), ("L5", "To five local"), ("L2w", "To two local weeks")):
        for offset, prefix, color in ((0.0, True, GD_COLOR), (0.38, False, OTHER_COLOR)):
            subset = milestones.loc[milestones["prefix44"].eq(prefix)]
            start = pd.to_numeric(subset["M0_week_index"], errors="coerce")
            end = pd.to_numeric(subset[f"{code}_week_index"], errors="coerce")
            values = (end - start).dropna().to_numpy(dtype=float)
            position = cursor + offset
            if len(values):
                ax_c.boxplot(
                    values,
                    positions=[position],
                    widths=0.28,
                    showfliers=False,
                    patch_artist=True,
                    medianprops={"color": "#272727", "linewidth": 0.8},
                    boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.6, "alpha": 0.35},
                    whiskerprops={"color": color, "linewidth": 0.6},
                    capprops={"color": color, "linewidth": 0.6},
                )
                jitter = rng.uniform(-0.07, 0.07, size=len(values))
                ax_c.scatter(np.full(len(values), position) + jitter, values, s=4, color=color, alpha=0.55, linewidths=0, zorder=3)
            sample_sizes.append(len(values))
        labels_c.append((cursor + 0.19, f"{label}\n{sample_sizes[-2]} / {sample_sizes[-1]}"))
        cursor += 1.15
    ax_c.set_xticks([item[0] for item in labels_c])
    ax_c.set_xticklabels([item[1] for item in labels_c])
    ax_c.set_ylabel("Weeks from first reported case")
    ax_c.set_xlim(-0.35, cursor - 0.45)
    ax_c.legend(
        handles=[
            Patch(facecolor=GD_COLOR, label="Guangdong Province"),
            Patch(facecolor=OTHER_COLOR, label="Other provinces"),
        ],
        loc="upper right",
        fontsize=6,
    )
    ax_c.text(0.0, -0.42, "Counts under each label are Guangdong Province / other provinces. Cities that have not reached the later milestone are omitted. Points are cities; the box is the interquartile range and the line is the median.", transform=ax_c.transAxes, ha="left", va="top", fontsize=5.5, color="#4D4D4D", clip_on=False)
    panel_letter(ax_c, "b")
    save_figure(fig, "fig3_expansion")


def k_cover(values: np.ndarray, proportion: float) -> float:
    total = float(np.sum(values))
    if total <= 0:
        return np.nan
    cumulative = np.cumsum(np.sort(values)[::-1])
    return float(int(np.searchsorted(cumulative, proportion * total - 1e-6, side="left") + 1))


def assert_k_cover() -> None:
    sample = np.array([10, 0, 5, 5])
    if k_cover(sample, 0.5) != 1 or k_cover(sample, 1.0) != 3 or k_cover(sample, 0.3) != 1:
        raise AssertionError("city-coverage definition failed its check")
    if not math.isnan(k_cover(np.zeros(3), 1.0)):
        raise AssertionError("zero-week coverage should be undefined")


def concentration_table(panel: pd.DataFrame, column: str, last_week: int) -> pd.DataFrame:
    wide = (
        panel.pivot(index="city_code", columns="week_index", values=column)
        .reindex(columns=list(range(1, last_week + 1)))
        .fillna(0)
        .to_numpy(dtype=float)
    )
    rows = []
    for position in range(last_week):
        values = wide[:, position]
        total = float(values.sum())
        row = {"week_index": position + 1, "cases": total, "leading_share": np.nan if total <= 0 else float(values.max() / total)}
        for level in P_LEVELS:
            row[f"k_{level}"] = k_cover(values, level / 100)
        rows.append(row)
    return pd.DataFrame(rows)


def figure_concentration(tables: dict[str, pd.DataFrame], column: str, stem: str, ylabel_cases: str) -> None:
    weeks = tables["weeks"]
    last_week = int(weeks["week_index"].max())
    result = concentration_table(tables["panel"], column, last_week)
    result = weeks[["week_index", "week_start"]].merge(result, on="week_index", how="left")
    fig = plt.figure(figsize=millimetres(183, 104))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.18)
    ax = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1], sharex=ax)
    x = result["week_index"].to_numpy()
    small = result["cases"].to_numpy() < 20
    zero = result["cases"].to_numpy() <= 0
    for level in P_LEVELS:
        y = result[f"k_{level}"].to_numpy(dtype=float)
        strong = np.where(small | zero, np.nan, y)
        light = np.where(~small | zero, np.nan, y)
        ax.plot(x, strong, color=P_COLORS[level], lw=1.05, label=f"{level}%")
        ax.plot(x, light, color="#BDBDBD", lw=0, marker="o", ms=2.4)
    ax.plot([], [], color="#BDBDBD", marker="o", lw=0, ms=3, label="<20 cases")
    ax.set_ylabel("Minimum number of cities")
    ax.legend(loc="upper left", ncol=6, fontsize=6)
    panel_letter(ax, "a")
    share = result["leading_share"].to_numpy(dtype=float)
    strong_share = np.where(small | zero, np.nan, share)
    light_share = np.where(~small | zero, np.nan, share)
    ax2.plot(x, strong_share, color="#272727", lw=1.0)
    ax2.plot(x, light_share, color="#BDBDBD", lw=0, marker="o", ms=2.6)
    ax2.set_ylabel("Share in the leading city")
    ax2.set_ylim(-0.03, 1.05)
    ax2.text(0.0, -0.32, f"Minimum cities whose cases cover the stated share of {ylabel_cases}. Light markers are weeks with fewer than 20 cases; weeks with none are omitted.", transform=ax2.transAxes, fontsize=5.5, color="#4D4D4D", clip_on=False)
    panel_letter(ax2, "b")
    style_time_axis(ax2, weeks)
    ax.tick_params(labelbottom=False)
    save_figure(fig, stem)


def figure_5(tables: dict[str, pd.DataFrame]) -> None:
    weeks = tables["weeks"]
    panel = tables["public_panel"].copy()
    panel["province_code"] = panel["city_code"].str[:2]
    categories = [key for key, _, _ in PUBLIC_SERIES]
    colors = {key: color for key, _, color in PUBLIC_SERIES}
    labels = {key: label for key, label, _ in PUBLIC_SERIES}
    province = panel.groupby("province_code", as_index=True)[categories].sum()
    totals = province.sum(axis=1).sort_values(ascending=True)
    pivot = province.reindex(totals.index)
    missing_names = sorted(set(pivot.index) - set(PROVINCE_NAMES))
    if missing_names:
        raise AssertionError(f"missing province names for codes: {missing_names}")
    weekly = panel.groupby("week_index")[categories].sum().reindex(
        list(range(1, int(weeks["week_index"].max()) + 1)), fill_value=0
    )
    fig = plt.figure(figsize=millimetres(183, 188))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.25, 1], hspace=0.30, top=0.91, bottom=0.07, left=0.17, right=0.91)
    ax = fig.add_subplot(grid[0])
    y = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for category in categories:
        height = pivot[category].to_numpy(dtype=float)
        if height.sum() == 0:
            continue
        ax.barh(y, height, left=bottom, height=0.72, color=colors[category], label=labels[category], linewidth=0)
        bottom += height
    ax.set_yticks(y)
    ax.set_yticklabels([PROVINCE_NAMES[code] for code in pivot.index], fontsize=6)
    ax.set_xlabel("Classified cases")
    ax.set_xscale("symlog", linthresh=10, linscale=0.8)
    ax.set_xlim(0, max(bottom) * 1.08)
    panel_letter(ax, "a")
    ax2 = fig.add_subplot(grid[1])
    line = ax2.twinx()
    week_total = weekly.sum(axis=1).to_numpy(dtype=float)
    shares = weekly.to_numpy(dtype=float)
    shares[week_total == 0] = 0
    positive = week_total > 0
    shares[positive] = shares[positive] / week_total[positive, None]
    bottom = np.zeros(len(weekly))
    x_week = weekly.index.to_numpy()
    for column, category in enumerate(categories):
        if weekly[category].sum() == 0:
            continue
        height = np.where(positive, shares[:, column], 0)
        ax2.bar(x_week, height, bottom=bottom, width=0.86, color=colors[category], linewidth=0)
        bottom += height
    line.plot(x_week, week_total, color="#272727", lw=0.9)
    line.set_ylabel("Weekly classified cases")
    line.spines["right"].set_visible(True)
    line.spines["top"].set_visible(False)
    ax2.set_ylabel("Share of weekly cases")
    ax2.set_ylim(0, 1)
    style_time_axis(ax2, weeks)
    panel_letter(ax2, "b")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.54, 0.975), ncol=3, fontsize=6, frameon=False)
    save_figure(fig, "fig5_source_composition")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw the data-descriptor figures from the public city-week release."
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE,
        help="Directory containing the public CSV files and release_summary.json.",
    )
    parser.add_argument(
        "--basemap-dir",
        type=Path,
        default=None,
        help=(
            "Directory with Tianditu 中国_省.geojson and 中国_市.geojson. "
            f"Download: {BASEMAP_DOWNLOAD}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES,
        help="Directory for the SVG, PDF and PNG figures.",
    )
    args = parser.parse_args()
    global FIGURES
    FIGURES = args.output_dir
    apply_style()
    assert_k_cover()
    assert_label_rules()
    tables = load_tables(args.release_dir)
    global WEEK_MIN, WEEK_MAX
    WEEK_MIN = int(tables["weeks"]["week_index"].min())
    WEEK_MAX = int(tables["weeks"]["week_index"].max())
    basemap_dir = resolve_basemap_dir(args.basemap_dir)
    basemap = load_basemap(basemap_dir / BASEMAP_FILES[0], basemap_dir / BASEMAP_FILES[1])
    assert_cities_match(basemap, tables["milestones"])
    figure_1(tables, basemap)
    figure_2(tables, basemap)
    figure_4(tables)
    figure_concentration(tables, "cases_total", "fig4_concentration", "weekly reported cases")
    figure_5(tables)
    print(json.dumps({
        "release_summary": {
            key: tables["summary"][key]
            for key in (
                "data_version",
                "rows",
                "cities",
                "weeks",
                "reported_cases",
                "inferred_imported_cases",
            )
        },
        "output_dir": str(FIGURES),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
