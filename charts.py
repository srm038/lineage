import copy
import datetime
import math
import os
from typing import Any, Dict, Iterable, List, Set, Tuple, Union
import bs4

from utils import (
    descent,
    getApproxBirth,
    getApproxVitals,
    getChildren,
    getMarriageYear,
    getParent,
    getSpouse,
    getState,
    getVitalYear,
)

from config import people, generations


def box_dim(N, f=2, W=48, m=1):
    """

    :param N: number of boxes
    :param f: fraction of w relative to spacing
    :param W: width/height of container
    :param m: margin
    :return: box width/height for given tree size
    """

    w = (W - 2 * m) / (N + (N - 1) / f)
    return round(w, 2), round(w / f, 2)


def generate_genealogy_tree(root, main=False):
    tree = {
        root: "parent{g{"
        + f"{people[root].get('name', {}).get('first')
             } {people[root].get('name', {}).get('last') or '---'}"
        + "}}"
    }
    main_people = set()
    for g in generations:
        for p in g:
            tree.update(
                {
                    p: "parent{g{"
                    + f"{people[p].get('name', {}).get('first')
                         } {people[p].get('name', {}).get('last') or '---'}"
                    + "}}"
                }
            )
            main_people.add(p)
    done = set()
    collapsed_tree = tree.copy()
    for g in generations:
        for c in sorted(
            set.union(*[people[p]["child"] for p in g]),
            key=lambda i: people[i]["gender"],
            reverse=True,
        ):
            if main and c not in main_people:
                continue
            if c in done:
                continue
            father, mother, siblings = "", "", ""
            if people[c]["father"] in g:
                if people[c]["father"] in done:
                    father = tree[people[c]["father"]]
                else:
                    father = collapsed_tree[people[c]["father"]]
                done.add(people[c]["father"])
            if people[c]["mother"] in g:
                if people[c]["mother"] in done:
                    mother = tree[people[c]["mother"]]
                else:
                    mother = collapsed_tree[people[c]["mother"]]
                done.add(people[c]["mother"])
                try:
                    for s in sorted(
                        people[people[c]["father"]]["children"][people[c]["mother"]],
                        key=lambda p: people[p]["gender"],
                        reverse=True,
                    ):
                        # for s in sorted(people[people[c]['father']]['children'][people[c]['mother']], key=lambda p: get_birth_year(p), reverse=True):
                        if s == c:
                            continue
                        if main and s not in main_people:
                            continue
                        siblings += "c{" + people[s]["shortname"] + "}"
                        done.add(s)
                except:
                    pass
            collapsed_tree[c] = (
                collapsed_tree[c][:-1]
                + f"{siblings}{father}{mother}"
                + collapsed_tree[c][-1:]
            )
    print(collapsed_tree[root])


def generateHTree(file: str, p: str, size: int = 90):
    N = len(generations) // 2
    spacing: int = size + int(size / 6)

    ancestors, descendants, initialPositions = getInitialPositionsH(size, N, p)
    compaction = [
        horizontalCompactionLTR,
        horizontalCompactionRTL,
        verticalCompactionTTB,
        verticalCompactionBTT,
    ]
    while True:
        oldPositions = copy.deepcopy(initialPositions)
        compaction = compaction[::-1]
        for outerEdge in [True, False]:
            for f in compaction:
                initialPositions = f(
                    initialPositions, descendants, ancestors, outerEdge
                )
                initialPositions = compactTwigsLeaves(
                    ancestors, descendants, initialPositions, spacing
                )
        if (
            getHArea(oldPositions, descendants)["area"]
            <= getHArea(initialPositions, descendants)["area"]
        ):
            initialPositions = copy.deepcopy(oldPositions)
            print(getHArea(initialPositions, descendants))
            break

    area = getHArea(initialPositions, descendants)
    print(area)
    drawHTree(area, descendants, file, initialPositions, size, p)


def compactTwigsLeaves(ancestors, descendants, initialPositions, spacing):
    initialPositions = verticalCompactionLeaves(
        initialPositions, descendants, ancestors, spacing
    )
    initialPositions = horizontalCompactionLeaves(
        initialPositions, descendants, ancestors, spacing
    )
    initialPositions = horizontalCompactionTwig(
        initialPositions, descendants, ancestors, spacing
    )
    initialPositions = verticalCompactionTwig(
        initialPositions, descendants, ancestors, spacing
    )
    return initialPositions


def verticalCompactionLeaves(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int
) -> Dict:
    bars = getYBars(ancestors, descendants, initialPositions)
    for b in bars:
        for node in bars[b]:
            if node not in ancestors and descendants[node] not in bars[b]:
                childDist = getChildDist(descendants, initialPositions, node)
                if childDist > spacing:
                    xp, yp = initialPositions[node]
                    yc = initialPositions[descendants[node]][1]
                    initialPositions[node] = (xp, yc + spacing * {0: -1, 1: 1}[yc < yp])
    return initialPositions


def horizontalCompactionLeaves(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int
) -> Dict:
    bars = getXBars(ancestors, descendants, initialPositions)
    for b in bars:
        for node in bars[b]:
            if node not in ancestors and descendants[node] not in bars[b]:
                childDist = getChildDist(descendants, initialPositions, node)
                if childDist > spacing:
                    xp, yp = initialPositions[node]
                    xc = initialPositions[descendants[node]][0]
                    initialPositions[node] = (xc + spacing * {0: -1, 1: 1}[xc < xp], yp)
    return initialPositions


def verticalCompactionTwig(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int
) -> Dict:
    bars = getYBars(ancestors, descendants, initialPositions)
    visibilityTTB = getVisibilityTTB(bars, initialPositions)
    visibilityBTT = getVisibilityBTT(bars, initialPositions)
    for b in bars:
        if (
            len(bars[b]) == 3
            and bars[b][0] not in ancestors
            and bars[b][-1] not in ancestors
        ):
            p = bars[b][1]
        elif len(bars[b]) == 2 and any(i not in ancestors for i in bars[b]):
            p = bars[b][0] if bars[b][1] not in ancestors else bars[b][1]
        else:
            continue
        childDist = getChildDist(descendants, initialPositions, p)
        if childDist == spacing:
            continue
        c = descendants[p]
        if c in bars.get(visibilityTTB.get(b, ()), []) + bars.get(
            visibilityBTT.get(b, ()), []
        ):
            yc = initialPositions[c][1]
            for node in bars[b]:
                xa, ya = initialPositions[node]
                initialPositions[node] = (xa, yc + spacing * {0: -1, 1: 1}[yc < ya])
    return initialPositions


def horizontalCompactionTwig(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int
) -> Dict:
    bars = getXBars(ancestors, descendants, initialPositions)
    visibilityLTR = getVisibilityLTR(bars, initialPositions)
    visibilityRTL = getVisibilityRTL(bars, initialPositions)
    for b in bars:
        if (
            len(bars[b]) == 3
            and bars[b][0] not in ancestors
            and bars[b][-1] not in ancestors
        ):
            p = bars[b][1]
        elif len(bars[b]) == 2 and any(i not in ancestors for i in bars[b]):
            p = bars[b][0] if bars[b][1] not in ancestors else bars[b][1]
        else:
            continue
        childDist = getChildDist(descendants, initialPositions, p)
        if childDist == spacing:
            continue
        c = descendants[p]
        if c in bars.get(visibilityLTR.get(b, ()), []) + bars.get(
            visibilityRTL.get(b, ()), []
        ):
            xc = initialPositions[c][0]
            for node in bars[b]:
                xa, ya = initialPositions[node]
                initialPositions[node] = (xc + spacing * {0: -1, 1: 1}[xc < xa], ya)
    return initialPositions


def drawHTree(area, descendants, file, initialPositions, size, p0):
    with open(rf"{os.getcwd()}\out\{file}-H.svg", "r") as f:
        svg = bs4.BeautifulSoup(f, "xml")
    treeStyle = f"fill:burlywood;"
    pStyle = f"stroke:black;stroke-width:2px;"

    def gen(g):
        return f"fill:white;opacity:{g / len(generations):0.2f};"

    nameStyle = "fill:black;text-anchor:middle;alignment-baseline:middle;font-size:9pt;font-family:Chomsky;"
    duplicate = "opacity:0.5;"
    linesStyle = "stroke:black;stroke-width:5px;"
    styles = (
        f"<style type='text/css'>"
        f".tree {{{treeStyle}}} "
        f".name {{{nameStyle}}} "
        f".duplicate {{{duplicate}}} "
        f".line {{{linesStyle}}} "
        f".tree {{{treeStyle}}}"
        f".primary {{{pStyle}}} "
        f"</style>"
    )
    tree = ""
    names = ""
    lines = ""
    minx = area["minx"]
    maxx = area["maxx"]
    miny = area["miny"]
    maxy = area["maxy"]
    for p in initialPositions:
        x, y = initialPositions[p]
        xc, yc = initialPositions[descendants[p]]
        key = p[:-2]
        rx = {"M": int(size / 6), "F": int(size / 2)}[people[key]["gender"]]
        tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' rx='{rx}' width='{
            size}' height='{size}' class='tree{' primary' if p0 == p[:-2] else ''}' id='{p}' />"
        tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' width='{
            size}' height='{size}' style='{gen(people[key]['generation'])}'/>"
        if people[key].get("name", {}).get("last"):
            names += f"<text class='name{' duplicate' if int(p[-1]) != 0 else ''}'><tspan x='{x}' y='{y}' dy='-2.5pt'>{people[key].get(
                'name', {}).get('first')}</tspan><tspan x='{x}' y='{y}' dy='7.5pt'>{people[key].get('last', '')}</tspan></text>"
        else:
            names += f"<text class='name{' duplicate' if int(p[-1]) != 0 else ''}'><tspan x='{
                x}' y='{y}'>{people[key].get('name', {}).get('first')}</tspan></text>"
        lines += f"<path d='M {x},{y} L {xc},{yc}' class='line'/>"
    svg.find("svg").clear()
    svg.find("svg").contents = bs4.BeautifulSoup(styles, "html.parser").contents
    svg.find("svg").contents += bs4.BeautifulSoup(lines, "html.parser").contents
    svg.find("svg").contents += bs4.BeautifulSoup(tree, "html.parser").contents
    svg.find("svg").contents += bs4.BeautifulSoup(names, "html.parser").contents
    svg.find("svg").attrs.update(
        {
            "viewBox": f"{minx - size / 2} {miny - size / 2} {maxx - minx + size} {maxy - miny + size}"
        }
    )
    with open(rf"{os.getcwd()}\out\{file}-H.svg", "w") as f:
        f.write(svg.prettify())


def getInitialPositionsH(s, N, p0) -> Tuple[Dict, Dict, Dict]:
    initialPositions = {}
    descendants = {}
    ancestors = {}
    done = set()
    current = {p0}
    while current:
        for c in list(current):
            personDescent = descent(c, p0)
            for d in personDescent:
                x, y = positionH(d, N, s=s + 15)
                key = "-".join([c, str(int(c in done))])
                initialPositions.update({key: (x, y)})
                xc, yc = positionH(d[1:], N)
                done.add(c)
                keyc = [i for i in initialPositions if initialPositions[i] == (xc, yc)]
                if not keyc:
                    continue
                keyc = keyc[0]
                descendants.update({key: keyc})
                ancestors.setdefault(keyc, set())
                ancestors[keyc].add(key)
            if people[c].get("father") in people:
                current.add(people[c]["father"])
            if people[c].get("mother") in people:
                current.add(people[c]["mother"])
            current.discard(c)
    return ancestors, descendants, initialPositions


def positionH(d, N, s=90 + 15) -> Tuple[int, int]:
    x, y = 0, 0
    if len(d) == 1:
        return x, y
    for i, j in enumerate(d[::-1][1:]):
        if not (i + 1) % 2:
            y += (
                (-1 if people[j]["gender"] == "M" else 1)
                * s
                * 2 ** (N - people[j]["generation"] // 2)
            )
        else:
            x += (
                (-1 if people[j]["gender"] == "M" else 1)
                * s
                * 2 ** (N - people[j]["generation"] // 2 - 1)
            )
    return x, y


def getHArea(initialPositions: Dict, descendants: Dict) -> Dict[str, int]:
    minx = min([p[0] for p in initialPositions.values()])
    maxx = max([p[0] for p in initialPositions.values()])
    miny = min([p[1] for p in initialPositions.values()])
    maxy = max([p[1] for p in initialPositions.values()])
    area = 0
    for p in initialPositions:
        if p not in descendants:
            continue
        edge = getChildDist(descendants, initialPositions, p)
        area += edge
    return {"maxx": maxx, "maxy": maxy, "minx": minx, "miny": miny, "area": area}


def getChildDist(descendants: Dict, initialPositions: Dict, p: str) -> int:
    c = descendants[p]
    xp, yp = initialPositions[p]
    xc, yc = initialPositions[c]
    edge = abs((xc - xp) + (yc - yp))
    return edge


def horizontalCompactionLTR(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, outerEdge: bool = False
) -> Dict:
    bars = getXBars(ancestors, descendants, initialPositions)
    # Create visibility graph
    # Update bar location
    spacing = 90 + 15
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True):
        visibility = getVisibilityLTR(bars, initialPositions)
        if b not in visibility:
            c = sorted(
                bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True
            )[0]
            x = initialPositions[bars[c][0]][0] + spacing
            if not outerEdge:
                continue
        else:
            c = visibility[b]
            x = initialPositions[bars[c][0]][0]
        for node in bars[b]:
            initialPositions[node] = (x - spacing, initialPositions[node][1])
    return initialPositions


def getVisibilityLTR(bars, initialPositions):
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0]):
        x = initialPositions[bars[b][0]][0]
        y1 = min([initialPositions[i][1] for i in bars[b]])
        y2 = max([initialPositions[i][1] for i in bars[b]])
        for c in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0]):
            if initialPositions[bars[c][0]][0] <= x:
                continue
            y3 = min([initialPositions[i][1] for i in bars[c]])
            y4 = max([initialPositions[i][1] for i in bars[c]])
            if y3 <= y1 <= y4 or y3 <= y2 <= y4:
                visibility[b] = c
                break
            if y1 <= y3 <= y2 or y1 <= y4 <= y2:
                visibility[b] = c
                break
    return visibility


def verticalCompactionBTT(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, outerEdge: bool = False
) -> Dict:
    bars = getYBars(ancestors, descendants, initialPositions)
    # Create visibility graph
    spacing = 90 + 15
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1]):
        visibility = getVisibilityBTT(bars, initialPositions)
        if b not in visibility:
            c = sorted(bars, key=lambda b: initialPositions[bars[b][0]][1])[0]
            y = initialPositions[bars[c][0]][1] - spacing
            if not outerEdge:
                continue
        else:
            c = visibility[b]
            y = initialPositions[bars[c][0]][1]
        for node in bars[b]:
            initialPositions[node] = (initialPositions[node][0], y + spacing)
    return initialPositions


def getVisibilityBTT(bars, initialPositions):
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True):
        y = initialPositions[bars[b][0]][1]
        x1 = min([initialPositions[i][0] for i in bars[b]])
        x2 = max([initialPositions[i][0] for i in bars[b]])
        for c in sorted(
            bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True
        ):
            if initialPositions[bars[c][0]][1] >= y:
                continue
            x3 = min([initialPositions[i][0] for i in bars[c]])
            x4 = max([initialPositions[i][0] for i in bars[c]])
            if x3 <= x1 <= x4 or x3 <= x2 <= x4:
                visibility[b] = c
                break
            if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                visibility[b] = c
                break
    return visibility


def horizontalCompactionRTL(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, outerEdge: bool = False
) -> Dict:
    bars = getXBars(ancestors, descendants, initialPositions)
    # Create visibility graph
    # Update bar location
    spacing = 90 + 15
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0]):
        visibility = getVisibilityRTL(bars, initialPositions)
        if b not in visibility:
            c = sorted(bars, key=lambda b: initialPositions[bars[b][0]][0])[0]
            x = initialPositions[bars[c][0]][0] - spacing
            if not outerEdge:
                continue
        else:
            c = visibility[b]
            x = initialPositions[bars[c][0]][0]
        for node in bars[b]:
            initialPositions[node] = (x + spacing, initialPositions[node][1])
    return initialPositions


def getVisibilityRTL(bars, initialPositions):
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True):
        x = initialPositions[bars[b][0]][0]
        y1 = min([initialPositions[i][1] for i in bars[b]])
        y2 = max([initialPositions[i][1] for i in bars[b]])
        for c in sorted(
            bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True
        ):
            if initialPositions[bars[c][0]][0] >= x:
                continue
            y3 = min([initialPositions[i][1] for i in bars[c]])
            y4 = max([initialPositions[i][1] for i in bars[c]])
            if y3 <= y1 <= y4 or y3 <= y2 <= y4:
                visibility[b] = c
                break
            if y1 <= y3 <= y2 or y1 <= y4 <= y2:
                visibility[b] = c
                break
    return visibility


def verticalCompactionTTB(
    initialPositions: Dict, descendants: Dict, ancestors: Dict, outerEdge: bool = False
) -> Dict:
    bars = getYBars(ancestors, descendants, initialPositions)
    spacing = 90 + 15
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True):
        visibility = getVisibilityTTB(bars, initialPositions)
        if b not in visibility:
            c = sorted(
                bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True
            )[0]
            y = initialPositions[bars[c][0]][1] + spacing
            if not outerEdge:
                continue
        else:
            c = visibility[b]
            y = initialPositions[bars[c][0]][1]
        for node in bars[b]:
            initialPositions[node] = (initialPositions[node][0], y - spacing)
    return initialPositions


def getVisibilityTTB(bars, initialPositions):
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1]):
        y = initialPositions[bars[b][0]][1]
        x1 = min([initialPositions[i][0] for i in bars[b]])
        x2 = max([initialPositions[i][0] for i in bars[b]])
        for c in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1]):
            if initialPositions[bars[c][0]][1] <= y:
                continue
            x3 = min([initialPositions[i][0] for i in bars[c]])
            x4 = max([initialPositions[i][0] for i in bars[c]])
            if x3 <= x1 <= x4 or x3 <= x2 <= x4:
                visibility[b] = c
                break
            if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                visibility[b] = c
                break
    return visibility


def getXBars(
    ancestors: Dict, descendants: Dict, initialPositions: Dict
) -> Dict[Tuple[int, int], List[str]]:
    xPositions: Set[int] = {initialPositions[i][0] for i in initialPositions}
    # Create axis dictionary
    axis: Dict = {}
    for x in sorted(xPositions):
        nodes: Iterable = sorted(
            {i for i in initialPositions if initialPositions[i][0] == x},
            key=lambda n: initialPositions[n][1],
        )
        axis.update({x: list(nodes)})
    # Create bar dictionary
    bars: Dict[Tuple[int, int], List[str]] = {}
    for x in axis:
        barIndex = 0
        done = set()
        bars.update({(x, barIndex): []})
        for i, u in enumerate(axis[x]):
            if u in done:
                continue
            bars[(x, barIndex)].append(u)
            if i + 1 == len(axis[x]):
                break
            v = axis[x][i + 1]
            if not v == descendants.get(u) and v not in ancestors.get(u, set()):
                barIndex += 1
                bars.update({(x, barIndex): []})
            done.add(u)
    return bars


def getYBars(
    ancestors: Dict, descendants: Dict, initialPositions: Dict
) -> Dict[Tuple[int, int], List[str]]:
    yPositions: Set[int] = {initialPositions[i][1] for i in initialPositions}
    # Create axis dictionary
    axis: Dict = {}
    for y in sorted(yPositions):
        nodes: Iterable = sorted(
            {i for i in initialPositions if initialPositions[i][1] == y},
            key=lambda n: initialPositions[n][0],
        )
        axis.update({y: list(nodes)})
    # Create bar dictionary
    bars: Dict[Tuple[int, int], List[str]] = {}
    for y in axis:
        barIndex = 0
        done = set()
        bars.update({(y, barIndex): []})
        for i, u in enumerate(axis[y]):
            if u in done:
                continue
            bars[(y, barIndex)].append(u)
            if i + 1 == len(axis[y]):
                break
            v = axis[y][i + 1]
            if not v == descendants.get(u) and v not in ancestors.get(u, set()):
                barIndex += 1
                bars.update({(y, barIndex): []})
            done.add(u)
    return bars


def generateLineTree(file, root):
    currentYear: int = datetime.datetime.now().year + 10
    done, unknownb, unknownd = getApproxVitals(root)

    numGenerations = len(generations)
    ancestors, descendants, initialPositions = getInitialPositionsLine(
        pt2px(10), numGenerations, root
    )
    positions = verticalCompactionLineTTB(
        done, initialPositions, descendants, ancestors
    )
    positions = verticalCompactionLineBTT(
        done, initialPositions, descendants, ancestors
    )
    positions = verticalCompactionLineTTB(
        done, initialPositions, descendants, ancestors
    )
    for p in done:
        done[p]["y"] = positions[p]
    minY = min(done[p]["y"] for p in done)
    for p in done:
        done[p]["y"] = done[p]["y"] - minY

    drawLineChart(currentYear, done, file, unknownb, unknownd)


def drawLineChart(
    currentYear: int, done: dict, file: str, unknownb: set, unknownd: set
):
    def addPt(i: str, event: str, pt: int = 0) -> float:
        return done[i][event] + pt / 0.75

    minx = min(done[p]["b"] for p in done) - 10
    maxx = currentYear
    widx = maxx - minx
    w = 18 * 96
    miny = min(done[p]["y"] for p in done) - 10
    maxy = max(done[p]["y"] for p in done) + 10
    with open(rf"{os.getcwd()}\out\{file}-lines.svg", "r") as f:
        svg = bs4.BeautifulSoup(f, "xml")
    lines = ""
    begats = ""
    names = ""
    years = ""
    flags = ""
    for p in done:
        if people[p].get("mother") in done:
            m = people[p]["mother"]
            begats += (
                f"<path d='M {done[p]['b'] * w /
                              widx:.3f},{addPt(p, 'y', -5):.3f} "
                f"V {addPt(m, 'y', 5):.3f}' "
                f"class='{'unbegat' if p in unknownb |
                          unknownd else 'begat'}' id='{m}-{p}'/>"
            )
            begats += f"<circle cx='{done[p]['b'] * w /
                                     widx:.3f}' cy='{addPt(m, 'y', -5):.3f}' r='3' />"
        if people[p].get("father") in done:
            f = people[p]["father"]
            begats += (
                f"<path d='M {done[p]['b'] * w /
                              widx:.3f},{addPt(p, 'y', 5):.3f} "
                f"V {addPt(f, 'y', -5):.3f}' "
                f"class='{'unbegat' if p in unknownb |
                          unknownd else 'begat'}' id='{f}-{p}'/>"
            )
            begats += f"<circle cx='{done[p]['b'] * w /
                                     widx:.3f}' cy='{addPt(f, 'y'):.3f}' r='3' />"
        if people[p].get("mother") in done or people[p].get("father") in done:
            begats += f"<circle cx='{done[p]['b'] * w /
                                     widx:.3f}' cy='{addPt(p, 'y'):.3f}' r='3' />"
    for p in done:
        if p in unknownb and p not in unknownd:
            p_class = "unknownb"
            if getState(p, "death"):
                flags += (
                    f"<image x='{
                        (done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' "
                    f"height='9pt' href='../flags/{
                        getState(p, 'death').lower()}.png'/>"
                )
        elif p not in unknownb and p in unknownd:
            p_class = "unknownd"
            if getState(p, "birth"):
                flags += (
                    f"<image x='{
                        (done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' "
                    f"height='9pt' href='../flags/{getState(p, 'birth').lower(
                    )}.png' style='transform: translateX(-15.72pt)'/>"
                )
        elif p in unknownb and p in unknownd:
            p_class = "unknownbd"
        else:
            p_class = "known"
            if getState(p, "death"):
                flags += (
                    f"<image x='{
                        (done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' "
                    f"height='9pt' href='../flags/{
                        getState(p, 'death').lower()}.png'/>"
                )
            if getState(p, "birth"):
                flags += (
                    f"<image x='{
                        (done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' "
                    f"height='9pt' href='../flags/{getState(p, 'birth').lower(
                    )}.png' style='transform: translateX(-15.72pt)'/>"
                )
        lines += (
            f"<rect x='{done[p]['b'] * w /
                        widx:.3f}' y='{addPt(p, 'y', -5):.3f}' "
            f"width='{(done[p]['d'] - done[p]['b']) * w /
                      widx:.3f}' height='{pt2px(10):.3f}' id='{p}'"
            f"class='{p_class}' />"
        )
        if p in unknownb:
            lines += (
                f"<rect x='{(done[p]['b'] - 5) * w /
                            widx:.3f}' y='{addPt(p, 'y', -5):.3f}' "
                f"width='{
                    6 * w / widx:.3f}' height='{pt2px(10):.3f}' id='{p}-b'"
                f"class='unknownbb' />"
            )
        if p in unknownd:
            lines += (
                f"<rect x='{(done[p]['d'] - 1) * w /
                            widx:.3f}' y='{addPt(p, 'y', -5):.3f}' "
                f"width='{
                    6 * w / widx:.3f}' height='{pt2px(10):.3f}' id='{p}-d'"
                f"class='unknownda' />"
            )
        for s in people[p].get("marriage", {}):
            if not getMarriageYear(people[p], s):
                continue
            flags += (
                f"<image x='{getMarriageYear(
                    people[p], s) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' "
                f"height='9pt' href='../flags/{getState(p, 'marriage')[s].lower(
                )}.png' style='transform: translateX(-7.36pt)'/>"
            )
    for p in done:
        names += (
            f"<text class='name {
                'nameunknown' if p in unknownb | unknownd else ''}'>"
            f"<tspan dx='{pt2px(2):.3f}' dy='{pt2px(1):.3f}' x={
                done[p]['b'] * w / widx:.3f} y={done[p]['y']:.3f}>"
            f"{people[p].get('name', {}).get('first')} {
                people[p].get('last', '')}</tspan></text>"
        )
    for y in range(math.floor(minx), math.ceil(maxx) + 1):
        if not y % 100:
            years += f"<path d='M {y * w /
                                   widx:.0f},{maxy} V {miny}' class='year' id='{y}'/>"
    minx = min((done[p]["b"] - 10) * w / widx for p in done)
    maxx = currentYear * w / widx
    svg.find("svg").find("g").clear()
    svg.find("svg").find("g").contents = bs4.BeautifulSoup(
        '<g id="yearLines"/><g id="begats"/><g id="lines"/><g id="flags" style="display:none"/><g id="names"/>',
        "html.parser",
    ).contents
    svg.find("g").find("g", id="yearLines").contents = bs4.BeautifulSoup(
        years, "html.parser"
    ).contents
    svg.find("g").find("g", id="begats").contents = bs4.BeautifulSoup(
        begats, "html.parser"
    ).contents
    svg.find("g").find("g", id="lines").contents = bs4.BeautifulSoup(
        lines, "html.parser"
    ).contents
    svg.find("g").find("g", id="flags").contents = bs4.BeautifulSoup(
        flags, "html.parser"
    ).contents
    svg.find("g").find("g", id="names").contents = bs4.BeautifulSoup(
        names, "html.parser"
    ).contents
    svg.find("svg").attrs.update(
        {"viewBox": f"{minx - 10} {miny - 10} {maxx - minx + 20} {maxy - miny + 20}"}
    )
    svg.find("svg").attrs.update(
        {"height": f"{maxy - miny}", "width": f"{maxx - minx}"}
    )
    svg.find("style").string = (
        ".known {fill:#888;} "
        ".unknownb {fill:url(#ub);} "
        ".unknownd {fill:url(#ud);} "
        ".unknownbd {fill:url(#ubd);} "
        ".unknownda {fill:url(#uda);} "
        ".unknownbb {fill:url(#ubb);} "
        ".begat {stroke-width:1pt; stroke:#888} "
        "#begats circle {fill:#888} "
        ".unbegat {stroke-width:1pt; stroke:#aaa} "
        ".name {dominant-baseline:middle; paint-order:stroke fill; font: bold 9pt Carlito; fill:black; stroke: white;stroke-width:2px} "
        ".nameunknown {fill:#333}"
        ".year {stroke-dasharray:5pt; stroke-width:2pt; stroke:#ccc}"
    )
    with open(rf"{os.getcwd()}\out\{file}-lines.svg", "w") as f:
        f.write(svg.prettify())


def getInitialPositionsLine(
    size: float, numGenerations: int, p0: str
) -> Tuple[Dict, Dict, Dict]:
    initialPositions: Dict[str, int] = {}
    descendants: Dict[str, str] = {}
    ancestors: Dict[Any, Any] = {}
    done: Set[str] = set()
    current: Set[str] = {p0}
    while current:
        for c in list(current):
            personDescent = descent(c, p0)[::-1]
            for descentLine in personDescent:
                y = positionLine(descentLine, numGenerations, size=size + 2)
                key = c  # '-'.join([c, str(int(c in done))])
                initialPositions.update({key: y})
                yc = positionLine(descentLine[1:], numGenerations, size=size + 2)
                done.add(c)
                keyc = [i for i in initialPositions if initialPositions[i] == yc]
                if not keyc:
                    continue
                keyc = keyc[0]
                descendants.update({key: keyc})
                ancestors.setdefault(keyc, set())
                ancestors[keyc].add(key)
            if people[c].get("father") in people:
                current.add(people[c]["father"])
            if people[c].get("mother") in people:
                current.add(people[c]["mother"])
            current.discard(c)
    return ancestors, descendants, initialPositions


def getVisibilityLineBTT(done: dict, bars: dict, positions: dict) -> dict:
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: positions[bars[b][0]], reverse=True):
        y = b[0]
        x1 = min([done[i]["b"] for i in bars[b]]) - 10
        x2 = max([done[i]["d"] for i in bars[b]]) + 10
        for c in sorted(bars, key=lambda b: positions[bars[b][0]], reverse=True):
            if c[0] >= y:
                continue
            x3 = min([done[i]["b"] for i in bars[c]]) - 10
            x4 = max([done[i]["d"] for i in bars[c]]) + 10
            if x3 <= x1 <= x4 or x3 <= x2 <= x4:
                visibility[b] = c
                break
            if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                visibility[b] = c
                break
    return visibility


def verticalCompactionLineBTT(
    done: dict,
    positions: Dict,
    descendants: Dict,
    ancestors: Dict,
    outerEdge: bool = False,
) -> Dict:
    bars = getYBarsLine(done, ancestors, descendants, positions)
    # Create visibility graph
    spacing = pt2px(15)
    for b in sorted(bars, key=lambda b: positions[bars[b][0]]):
        visibility = getVisibilityLineBTT(done, bars, positions)
        if b not in visibility:
            c = sorted(bars, key=lambda b: positions[bars[b][0]])[0]
            y = positions[bars[c][0]] - spacing
            if not outerEdge:
                continue
        else:
            c = visibility[b]
            y = positions[bars[c][0]]
        for node in bars[b]:
            positions[node] = y + spacing
    return positions


def verticalCompactionLineTTB(
    done: dict,
    positions: Dict,
    descendants: Dict,
    ancestors: Dict,
    outerEdge: bool = False,
) -> Dict:
    bars = getYBarsLine(done, ancestors, descendants, positions)
    # Create visibility graph
    spacing = pt2px(15)
    for b in sorted(bars, key=lambda b: positions[bars[b][0]], reverse=True):
        visibility = getVisibilityLineTTB(done, bars, positions)
        if b not in visibility:
            c = sorted(bars, key=lambda b: positions[bars[b][0]], reverse=True)[0]
            y = positions[bars[c][0]] + spacing
            if not outerEdge:
                continue
        else:
            c = visibility[b]
            y = positions[bars[c][0]]
        for node in bars[b]:
            positions[node] = y - spacing
    return positions


def getVisibilityLineTTB(done: dict, bars: dict, positions: dict) -> dict:
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: positions[bars[b][0]]):
        y = b[0]
        x1 = min([done[i]["b"] for i in bars[b]]) - 10
        x2 = max([done[i]["d"] for i in bars[b]]) + 10
        for c in sorted(bars, key=lambda b: positions[bars[b][0]]):
            if c[0] <= y:
                continue
            x3 = min([done[i]["b"] for i in bars[c]]) - 10
            x4 = max([done[i]["d"] for i in bars[c]]) + 10
            if x3 <= x1 <= x4 or x3 <= x2 <= x4:
                visibility[b] = c
                break
            if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                visibility[b] = c
                break
    return visibility


def getYBarsLine(
    done: Dict, ancestors: Dict, descendants: Dict, initialPositions: Dict
) -> Dict[Tuple[int, int], List[str]]:
    yPositions: Set[int] = {initialPositions[i] for i in initialPositions}
    # Create axis dictionary
    axis: Dict = {}
    for y in sorted(yPositions):
        nodes: Iterable = sorted(
            {i for i in initialPositions if initialPositions[i] == y},
            key=lambda n: done[n]["b"],
        )
        axis.update({y: list(nodes)})
    # Create bar dictionary
    bars: Dict[Tuple[int, int], List[str]] = {}
    for y in axis:
        barIndex = 0
        checked = set()
        bars.update({(y, barIndex): []})
        for i, u in enumerate(axis[y]):
            if u in checked:
                continue
            bars[(y, barIndex)].append(u)
            if i + 1 == len(axis[y]):
                break
            v = axis[y][i + 1]
            if not v == descendants.get(u) and v not in ancestors.get(u, set()):
                barIndex += 1
                bars.update({(y, barIndex): []})
            checked.add(u)
    return bars


def positionLine(descentList: list, numGenerations: int, size: float) -> int:
    y: int = 0
    if len(descentList) == 1:
        return 0
    for i, j in enumerate(descentList[::-1][1:]):
        yy = round(
            (1 if people[j]["gender"] == "F" else -1)
            * size
            * 2 ** (numGenerations - people[j]["generation"])
        )
        y += yy
    return y


def pt2px(pt: Union[float, int]) -> float:
    return pt / 0.75


def toPt(pt):
    return pt / 0.75


def drawZegelchart(p: str):
    currentYear: int = datetime.datetime.now().year
    descendants = generateZegelchart(p)

    def width(d):
        return (d["d"] or 2023) - d["b"]

    def y(i):
        return (2 * i) * toPt(10)

    xMin = min(d["b"] for d in descendants if d["b"]) - 10
    xMax = currentYear + 10
    xWidth = xMax - xMin
    yMin = 0
    yMax = len(descendants) * 2
    with open(rf"{os.getcwd()}\out\{p}-zegelchart.svg", "r") as f:
        svg = bs4.BeautifulSoup(f, "xml")
    lives = ""
    lines = ""
    names = ""
    done = set()
    for i, d in enumerate(descendants):
        if not d["b"]:
            continue
        lives += f"<rect id='{d['id']}' height='10pt' width='{
            width(d)}' x='{d['b']:.3f}' y='{y(i):.3f}' />"
        for s in getSpouse(d["id"]):
            if s not in done:
                sidx = next(
                    (j for j, e in enumerate(descendants) if e["id"] == s), None
                )
                lines += f"<path id='{d['id']}-{s}' d='M {getMarriageYear(
                    people[d['id']], s):.3f} {y(i) + toPt(5):.3f} V {y(sidx) + toPt(5):.3f}' />"
        if (
            next(
                (
                    j
                    for j, e in enumerate(descendants)
                    if e["id"] == getParent(d["id"], "father")
                ),
                None,
            )
            is not None
        ):
            if marriageYear := getMarriageYear(
                people[getParent(d["id"], "father")], getParent(d["id"], "mother")
            ):
                lines += f"<path id='{d['id']}-parent' d='M {d['b']
                    :.3f} {y(i) + toPt(5):.3f} H {marriageYear:.3f}' />"
        names += f"<text x='{currentYear + toPt(2):.3f}' y='{y(i) + toPt(5):.3f}'>{
            people[d['id']]['shortname']}</text>"
        done.add(d["id"])
    svg.find("g", id="lives").contents = bs4.BeautifulSoup(
        lives, "html.parser"
    ).contents
    svg.find("g", id="lines").contents = bs4.BeautifulSoup(
        lines, "html.parser"
    ).contents
    svg.find("g", id="names").contents = bs4.BeautifulSoup(
        names, "html.parser"
    ).contents
    svg.find("svg").attrs.update(
        {
            "viewBox": f"{xMin - 10:.3f} {yMin - 10:.3f} {xWidth + 20:.3f} {(yMax - yMin) * toPt(10) + 10:.3f}"
        }
    )
    with open(rf"{os.getcwd()}\out\{p}-zegelchart.svg", "w") as f:
        f.write(svg.prettify())


def generateZegelchart(p: str) -> List[Dict[str, Union[int, str, None]]]:
    def essentials(p: str):
        return {
            "id": p,
            "b": getVitalYear(p, "birth") or getApproxBirth(dict(), p),
            "d": getVitalYear(p, "death"),
        }

    descent = [essentials(p)]
    spouses = getSpouse(p)
    for spouse in spouses:
        if spouse:
            if people[p]["gender"] == "F":
                descent.insert(-2, essentials(spouse))
            else:
                descent.append(essentials(spouse))
        for child in getChildren(p, spouse):
            childDescent = generateZegelchart(child)
            descent[-1:-1] = childDescent
    return descent


def phi(a, b):
    return a + b / 2


def getRadialChartLayout(root: str) -> dict:
    approxVitals = getApproxVitals(root)
    radialChart: dict[
        str, dict[str, list[str] | int | float] | dict[str, list[str] | int | float]
    ] = {
        root: {
            "r1": approxVitals[root]["d"] - approxVitals[root]["d"],
            "r2": approxVitals[root]["d"] - approxVitals[root]["b"],
            "a": 0,
            "b": 180,
            "p": list(
                filter(None, [getParent(root, "father"), getParent(root, "mother")])
            ),
        }
    }
    radialChart[root].update(
        {"phi": phi(radialChart[root]["a"], radialChart[root]["b"])}
    )
    lenRadialChart = 0
    while lenRadialChart < len(radialChart):
        lenRadialChart = len(radialChart)
        for r in list(radialChart):
            d = radialChart[r]["b"] / 2
            radialChart[r].update({"d": d})
            for p in radialChart[r]["p"]:
                if p not in radialChart:
                    radialChart.update(
                        {
                            p: {
                                "r1": approxVitals[root]["d"] - approxVitals[p]["d"],
                                "r2": approxVitals[root]["d"] - approxVitals[p]["b"],
                                "a": radialChart[r]["a"]
                                + (1 if people[p]["gender"] == "M" else 0) * d,
                                "b": d,
                                "p": list(
                                    filter(
                                        None,
                                        [
                                            getParent(p, "father"),
                                            getParent(p, "mother"),
                                        ],
                                    )
                                ),
                            }
                        }
                    )
                    radialChart[p].update(
                        {"phi": phi(radialChart[p]["a"], radialChart[p]["b"])}
                    )
    return radialChart


def pol2xy(r, phi) -> Tuple[float, float]:
    return r * math.cos(math.radians(phi)), r * math.sin(math.radians(phi))


def dist(r1, phi1, r2, phi2) -> float:
    c1 = pol2xy(r1, phi1)
    c2 = pol2xy(r2, phi2)
    x = c2[0] - c1[0]
    y = c2[1] - c1[1]
    return math.hypot(x, y)


def bearing(r1, phi1, r2, phi2) -> float:
    x1, y1 = pol2xy(r1, phi1)
    x2, y2 = pol2xy(r2, phi2)
    return (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 360) % 360


def forceDirectRadialChart(radialChart, ks=1, kr=1):
    initialR = {}
    for m in radialChart:
        for p in radialChart[m]["p"]:
            r0 = (
                2
                * radialChart[m]["r2"]
                * math.sin(
                    math.radians(abs(radialChart[m]["phi"] - radialChart[p]["phi"]) / 2)
                )
            )
            initialR.setdefault(m, dict())
            initialR[m].update({p: r0})
    vector = {}
    for m in radialChart:
        vector.setdefault(m, [])
        # repulsive forces
        for n in radialChart:
            if m == n:
                continue
            r = dist(
                radialChart[m]["r2"],
                radialChart[m]["phi"],
                radialChart[n]["r2"],
                radialChart[n]["phi"],
            )
            b = bearing(
                radialChart[m]["r2"],
                radialChart[m]["phi"],
                radialChart[n]["r2"],
                radialChart[n]["phi"],
            )
            vector[m] += [(-kr / r**2, b)]
        # attractive forces
        for p in radialChart[m]["p"]:
            r0 = initialR[m][p]
            r = dist(
                radialChart[m]["r2"],
                radialChart[m]["phi"],
                radialChart[m]["r2"],
                radialChart[p]["phi"],
            )
            b = bearing(
                radialChart[m]["r2"],
                radialChart[m]["phi"],
                radialChart[m]["r2"],
                radialChart[p]["phi"],
            )
            vector[m] += [(ks * (r - r0), b)]
            vector.setdefault(p, [])
            vector[p] += [(ks * (r - r0), (b + 180) % 360)]
    for m in radialChart:
        x, y = pol2xy(radialChart[m]["r2"], radialChart[m]["phi"])
        x += sum([pol2xy(*f)[0] for f in vector[m]])
        y += sum([pol2xy(*f)[1] for f in vector[m]])
        radialChart[m]["phi"] = math.degrees(math.atan2(y, x))
    return radialChart


def distributeRadialChart(radialChart):
    sortedRadialChart = sorted(radialChart.keys(), key=lambda r: radialChart[r]["phi"])
    d = len(sortedRadialChart)
    for i, r in enumerate(sortedRadialChart):
        radialChart[r]["phi"] = 180 * i / (d - 1)
    return radialChart


def drawRadialChart(file, radialChart):
    with open(rf"{os.getcwd()}\out\{file}-radial.svg", "r") as f:
        svg = bs4.BeautifulSoup(f, "xml")
    paths = ""
    xMin, xMax, yMin, yMax = 0, 0, 0, 0
    for r in radialChart:
        paths += f'<path id="{r}" class="life {people[r]["gender"]}" d="M {radialChart[r]["r1"]} 0 H {
            radialChart[r]["r2"]}" transform="rotate({-radialChart[r]["phi"]} 0 0)" />'
        x, y = radialChart[r]["r2"] * math.cos(
            math.radians(radialChart[r]["phi"])
        ), radialChart[r]["r2"] * math.sin(math.radians(radialChart[r]["phi"]))
        xMin = min(xMin, x)
        xMax = max(xMax, x)
        yMin = min(yMin, -y)
        yMax = max(yMax, -y)
        for p in radialChart[r]["p"]:
            x1, y1 = pol2xy(radialChart[r]["r2"], -radialChart[r]["phi"])
            x2, y2 = pol2xy(radialChart[r]["r2"], -radialChart[p]["phi"])
            paths += f'<path id="{r}-{p}" class="parents" d="M {x1:.5f} {y1:.5f} A {radialChart[r]["r2"]} {
                radialChart[r]["r2"]} 0 0 {int(radialChart[r]["phi"] > radialChart[p]["phi"])} {x2:.5f} {y2:.5f}" />'
    svg.find("g", id="radial").contents = bs4.BeautifulSoup(
        paths, "html.parser"
    ).contents
    svg.find("svg").attrs.update(
        {
            "viewBox": f"{xMin - 10:.3f} {yMin - 10:.3f} {xMax - xMin + 20:.3f} {0 - yMin + 20:.3f}",
            "width": f"{xMax - xMin + 20:.3f}",
            "height": f"{0 - yMin + 20:.3f}",
        }
    )
    with open(rf"{os.getcwd()}\out\{file}-radial.svg", "w") as f:
        f.write(svg.prettify())
