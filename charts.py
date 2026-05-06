import copy
import datetime
import itertools
import math
import os
from typing import Any, Dict, Iterable, List, Set, Tuple, Union
import bs4

from utils import (
    descent,
    getApproxBirth,
    getApproxVitals,
    getChildren,
    getParent,
    getSpouse,
    getState,
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
        + f"{people[root].name.first} {people[root].name.last or '---'}"
        + "}}"
    }
    main_people = set()
    for g in generations:
        for p in g:
            tree.update(
                {
                    p: "parent{g{"
                    + f"{people[p].name.first} {people[p].name.last or '---'}"
                    + "}}"
                }
            )
            main_people.add(p)
    done = set()
    collapsed_tree = tree.copy()
    for g in generations:
        for c in sorted(
            set.union(*[people[p].child for p in g]),
            key=lambda i: people[i].gender,
            reverse=True,
        ):
            if main and c not in main_people:
                continue
            if c in done:
                continue
            father, mother, siblings = "", "", ""
            if people[c].father in g:
                if people[c].father in done:
                    father = tree[people[c].father]
                else:
                    father = collapsed_tree[people[c].father]
                done.add(people[c].father)
            if people[c].mother in g:
                if people[c].mother in done:
                    mother = tree[people[c].mother]
                else:
                    mother = collapsed_tree[people[c].mother]
                done.add(people[c].mother)
                try:
                    for s in sorted(
                        people[people[c].father].marriage[people[c].mother].children,
                        key=lambda p: people[p].gender,
                        reverse=True,
                    ):
                        # for s in sorted(people[people[c]['father']]['children'][people[c]['mother']], key=lambda p: get_birth_year(p), reverse=True):
                        if s == c:
                            continue
                        if main and s not in main_people:
                            continue
                        siblings += "c{" + people[s].name.shortname + "}"
                        done.add(s)
                except:
                    pass
            collapsed_tree[c] = (
                collapsed_tree[c][:-1]
                + f"{siblings}{father}{mother}"
                + collapsed_tree[c][-1:]
            )
    print(collapsed_tree[root])


def hasCollisionAtPosition(positions, node, spacing):
    node_pos = positions[node]
    node_x, node_y = node_pos

    for other_node, (other_x, other_y) in positions.items():
        if other_node == node:
            continue  # Skip self

        # Check if positions are too close (indicating overlap)
        if (
            abs(node_x - other_x) < spacing * 0.8
            and abs(node_y - other_y) < spacing * 0.8
        ):
            return True  # Collision detected

    return False


def countDescendantsNode(node, ancestors):
    if node not in ancestors or not ancestors[node]:
        return 0

    count = 0
    for child in ancestors[node]:
        count += 1 + countDescendantsNode(child, ancestors)
    return count


def findLinearChain(start_node, positions, descendants, ancestors):
    chain = [start_node]
    current = start_node

    # Extend forward (toward children) as long as each node has exactly one child
    while current in ancestors and len(ancestors[current]) == 1:
        child = list(ancestors[current])[0]
        if child in positions:
            chain.append(child)
            current = child
        else:
            break

    return chain


def normalize_positions(positions):
    """
    Normalize positions to bring coordinates into a manageable range
    while preserving relative distances between nodes
    """
    if not positions:
        return positions

    # Find the minimum coordinates
    min_x = min(pos[0] for pos in positions.values())
    min_y = min(pos[1] for pos in positions.values())

    # Shift all positions so that minimum becomes (0, 0) or close to it
    normalized_positions = {}
    for key, (x, y) in positions.items():
        normalized_positions[key] = (x - min_x, y - min_y)

    return normalized_positions


def find_empty_spaces(positions, descendants, ancestors, spacing):
    """
    Find large empty rectangular spaces in the layout that could be filled
    """
    if not positions:
        return []

    # Get bounding box
    min_x = min(pos[0] for pos in positions.values())
    max_x = max(pos[0] for pos in positions.values())
    min_y = min(pos[1] for pos in positions.values())
    max_y = max(pos[1] for pos in positions.values())

    # Create a grid to mark occupied spaces
    occupied = set(positions.values())

    # Look for empty rectangles of sufficient size
    empty_spaces = []

    # Check for horizontal gaps
    x_coords = sorted(set(pos[0] for pos in positions.values()))
    for i in range(len(x_coords) - 1):
        gap_start = x_coords[i]
        gap_end = x_coords[i + 1]
        if gap_end - gap_start > spacing:
            # Found a horizontal gap, check vertical extent
            y_coords = sorted(set(pos[1] for pos in positions.values()))
            for j in range(len(y_coords)):
                # Check if this region is mostly empty
                empty_count = 0
                total_check = 0
                for x in range(int(gap_start), int(gap_end), int(spacing / 2)):
                    for y in range(
                        int(y_coords[j]), int(y_coords[j] + spacing), int(spacing / 2)
                    ):
                        if (x, y) not in occupied:
                            empty_count += 1
                        total_check += 1

                if (
                    total_check > 0 and empty_count / total_check > 0.5
                ):  # More than 50% empty
                    empty_spaces.append(
                        {
                            "type": "horizontal_gap",
                            "x_range": (gap_start, gap_end),
                            "y_pos": y_coords[j],
                            "size": (gap_end - gap_start) * spacing,
                        }
                    )

    # Similar logic for vertical gaps
    y_coords = sorted(set(pos[1] for pos in positions.values()))
    for i in range(len(y_coords) - 1):
        gap_start = y_coords[i]
        gap_end = y_coords[i + 1]
        if gap_end - gap_start > spacing:
            x_coords = sorted(set(pos[0] for pos in positions.values()))
            for j in range(len(x_coords)):
                # Check if this region is mostly empty
                empty_count = 0
                total_check = 0
                for x in range(
                    int(x_coords[j]), int(x_coords[j] + spacing), int(spacing / 2)
                ):
                    for y in range(int(gap_start), int(gap_end), int(spacing / 2)):
                        if (x, y) not in occupied:
                            empty_count += 1
                        total_check += 1

                if (
                    total_check > 0 and empty_count / total_check > 0.5
                ):  # More than 50% empty
                    empty_spaces.append(
                        {
                            "type": "vertical_gap",
                            "y_range": (gap_start, gap_end),
                            "x_pos": x_coords[j],
                            "size": (gap_end - gap_start) * spacing,
                        }
                    )

    return sorted(empty_spaces, key=lambda x: x["size"], reverse=True)


def attempt_space_filling_move(
    positions, descendants, ancestors, empty_spaces, spacing
):
    """
    Attempt to move groups of nodes to fill empty spaces, respecting orthogonal edge constraints
    """
    if not empty_spaces:
        return positions

    new_positions = positions.copy()

    # For now, focus on the largest empty space
    largest_empty = empty_spaces[0]

    if largest_empty["type"] == "horizontal_gap":
        # For horizontal gaps, we can only shift nodes horizontally
        # We need to ensure that parent-child connections remain orthogonal
        gap_start = largest_empty["x_range"][0]

        # Find connected components that can be moved together
        # A connected component is a group of nodes that must move together
        # to preserve parent-child relationships
        processed = set()
        for node in positions:
            if node in processed:
                continue

            # Find the connected subtree that can move together
            # This is complex - for now, we'll use a simpler approach
            # Only move nodes that are to the right of the gap
            nodes_to_the_right = [
                k for k, (x, y) in positions.items() if x >= gap_start
            ]

            if nodes_to_the_right:
                # Check if moving these nodes left would violate constraints
                # For each node, check if its parent/child relationships would be preserved
                valid_move = True
                min_x_in_group = min(positions[k][0] for k in nodes_to_the_right)

                # Calculate maximum possible shift without violating constraints
                max_shift = min_x_in_group - gap_start

                if max_shift > 0:
                    # Apply the shift to all nodes in the group
                    for k in nodes_to_the_right:
                        old_x, old_y = new_positions[k]
                        new_positions[k] = (old_x - max_shift, old_y)
                    break  # Only process the first valid group for now

    elif largest_empty["type"] == "vertical_gap":
        # For vertical gaps, we can only shift nodes vertically
        gap_start = largest_empty["y_range"][0]

        nodes_below_gap = [k for k, (x, y) in positions.items() if y >= gap_start]

        if nodes_below_gap:
            # Calculate maximum possible shift
            min_y_in_group = min(positions[k][1] for k in nodes_below_gap)
            max_shift = min_y_in_group - gap_start

            if max_shift > 0:
                # Apply the shift to all nodes in the group
                for k in nodes_below_gap:
                    old_x, old_y = new_positions[k]
                    new_positions[k] = (old_x, old_y - max_shift)
                # Only process the first valid group for now
                pass

    return new_positions


def generateHTree(file: str, p: str, size: int = 90):
    N = len(generations) // 2 + 1
    spacing: int = size + int(size / 6)

    ancestors, descendants, initialPositions = getInitialPositionsH(size, N, p)

    print(getHArea(initialPositions, descendants))

    ordered_nodes = orderNodesByAncestors(ancestors, initialPositions)

    positions = normalize_positions(initialPositions)
    max_iterations = 5
    i = 0

    while i < max_iterations:
        prev_positions = normalize_positions(positions)

        moved = False

        for node in ordered_nodes:
            if node not in descendants:
                pass
            else:
                target = descendants[node]
                if target in positions:
                    moved |= moveNodeAndAncestorsTowardsDescendant(
                        positions,
                        node,
                        target,
                        ancestors,
                        descendants,
                        spacing,
                    )

        current_area = getHArea(positions, descendants)
        prev_area = getHArea(prev_positions, descendants)
        positions_changed = current_area["edge"] != prev_area["edge"]

        if not moved and not positions_changed:
            break

        print(
            i,
            getHArea(normalize_positions(positions), descendants),
            moved,
            positions_changed,
        )
        drawHTree(
            current_area, descendants, file, normalize_positions(positions), size, p
        )
        i += 1

    positions = normalize_positions(positions)
    print(area := getHArea(positions, descendants))
    drawHTree(area, descendants, file, positions, size, p)
    return ancestors, descendants, positions, ordered_nodes


def orderNodesByAncestors(ancestors, positions):
    """Order nodes by number of ancestors (upstream nodes), leaves first"""

    countAncestors = lambda node: (
        len(getAncestorsH(node, ancestors)) if node in ancestors else 0
    )

    # Sort nodes by number of ancestors (ascending - leaves first)
    return sorted(positions.keys(), key=countAncestors)


def positionsEqual(pos1, pos2):
    """Check if two position dictionaries are approximately equal"""
    if set(pos1.keys()) != set(pos2.keys()):
        return False

    for key in pos1:
        if (
            abs(pos1[key][0] - pos2[key][0]) > 0.1
            or abs(pos1[key][1] - pos2[key][1]) > 0.1
        ):
            return False
    return True


def moveNodeTowardsTarget(
    positions,
    node,
    target,
    spacing,
    x_bars,
    y_bars,
    ltr_visibility,
    rtl_visibility,
    ttb_visibility,
    btt_visibility,
):
    """Move a node towards a target (ancestor or descendant) considering visibility constraints"""
    if node not in positions or target not in positions:
        return False

    node_x, node_y = positions[node]
    target_x, target_y = positions[target]

    # Calculate desired movement direction
    dx = target_x - node_x
    dy = target_y - node_y

    # Get the bars this node belongs to
    node_x_bar = None
    node_y_bar = None

    for x_bar_key, nodes_in_bar in x_bars.items():
        if node in nodes_in_bar:
            node_x_bar = x_bar_key
            break

    for y_bar_key, nodes_in_bar in y_bars.items():
        if node in nodes_in_bar:
            node_y_bar = y_bar_key
            break

    new_x, new_y = node_x, node_y
    moved = False

    # Determine which direction has greater distance and move in that direction primarily
    if (
        abs(dx) >= abs(dy) and abs(dx) > 0.1
    ):  # Horizontal movement is larger or only movement
        # Calculate how far we can move horizontally based on visibility
        max_horizontal_move = min(abs(dx), spacing)  # Allow full spacing movement

        if node_x_bar:
            # Check LTR visibility
            if node_x_bar in ltr_visibility:
                visible_bar = ltr_visibility[node_x_bar]
                if visible_bar in x_bars:
                    visible_x = visible_bar[0]
                    if dx > 0:  # Want to move right
                        max_horizontal_move = min(
                            max_horizontal_move, max(0, visible_x - node_x - spacing)
                        )
                    else:  # Want to move left
                        max_horizontal_move = min(
                            max_horizontal_move, max(0, node_x - visible_x - spacing)
                        )

            # Check RTL visibility
            if node_x_bar in rtl_visibility:
                visible_bar = rtl_visibility[node_x_bar]
                if visible_bar in x_bars:
                    visible_x = visible_bar[0]
                    if dx > 0:  # Want to move right
                        max_horizontal_move = min(
                            max_horizontal_move, max(0, visible_x - node_x - spacing)
                        )
                    else:  # Want to move left
                        max_horizontal_move = min(
                            max_horizontal_move, max(0, node_x - visible_x - spacing)
                        )

        # Apply horizontal movement if possible
        if max_horizontal_move > 0:
            move_amount = min(abs(dx), max_horizontal_move)
            if dx > 0:  # Move right
                new_x = node_x + move_amount
            else:  # Move left
                new_x = node_x - move_amount

            if abs(new_x - node_x) > 0.1:  # Only move if there's a significant change
                moved = True
    elif abs(dy) > 0.1:  # Vertical movement is larger
        # Calculate how far we can move vertically based on visibility
        max_vertical_move = min(abs(dy), spacing)  # Allow full spacing movement

        if node_y_bar:
            # Check TTB visibility
            if node_y_bar in ttb_visibility:
                visible_bar = ttb_visibility[node_y_bar]
                if visible_bar in y_bars:
                    visible_y = visible_bar[0]
                    if dy > 0:  # Want to move down
                        max_vertical_move = min(
                            max_vertical_move, max(0, visible_y - node_y - spacing)
                        )
                    else:  # Want to move up
                        max_vertical_move = min(
                            max_vertical_move, max(0, node_y - visible_y - spacing)
                        )

            # Check BTT visibility
            if node_y_bar in btt_visibility:
                visible_bar = btt_visibility[node_y_bar]
                if visible_bar in y_bars:
                    visible_y = visible_bar[0]
                    if dy > 0:  # Want to move down
                        max_vertical_move = min(
                            max_vertical_move, max(0, visible_y - node_y - spacing)
                        )
                    else:  # Want to move up
                        max_vertical_move = min(
                            max_vertical_move, max(0, node_y - visible_y - spacing)
                        )

        # Apply vertical movement if possible
        if max_vertical_move > 0:
            move_amount = min(abs(dy), max_vertical_move)
            if dy > 0:  # Move down
                new_y = node_y + move_amount
            else:  # Move up
                new_y = node_y - move_amount

            if abs(new_y - node_y) > 0.1:  # Only move if there's a significant change
                moved = True

    # Actually update the position if movement occurred
    if moved:
        positions[node] = (new_x, new_y)

    return moved


def getAncestorsH(node, ancestors):
    """Recursively get all ancestors of a node"""
    all_ancestors = set()
    nodes_to_check = list(ancestors.get(node, set()))

    while nodes_to_check:
        current = nodes_to_check.pop()
        if current not in all_ancestors:
            all_ancestors.add(current)
            # Add the ancestors of this node to the list to check
            nodes_to_check.extend(ancestors.get(current, set()))

    return all_ancestors


def moveNodeAndAncestorsTowardsDescendant(
    positions: Dict,
    node: str,
    target: str,
    ancestors: Dict,
    descendants: Dict,
    spacing: int,
) -> bool:
    """Move a node and its ancestors toward target with direct distance calculation"""
    if node not in positions or target not in positions:
        return False

    nodeX, nodeY = positions[node]
    targetX, targetY = positions[target]

    dx = targetX - nodeX
    dy = targetY - nodeY

    isHorizPrimary = abs(dx) >= abs(dy)

    ancestralBlock = getAncestorsH(node, ancestors)
    nodesToCheck = {node}
    nodesToCheck.update(ancestralBlock)

    dir = 1 if (dx if isHorizPrimary else dy) > 0 else -1
    maxDistance = abs(dx if isHorizPrimary else dy)
    maxUnits = int(maxDistance // spacing)

    if maxUnits <= 0:
        return False

    static = {p: v for p, v in positions.items() if p not in nodesToCheck}
    barsX = getXBars(ancestors, descendants, static)
    barsY = getYBars(ancestors, descendants, static)

    px = {n: positions[n][0] for n in nodesToCheck}
    py = {n: positions[n][1] for n in nodesToCheck}

    minUnits = maxUnits

    if isHorizPrimary:
        # Moving horizontally - check vertical bars and nodes
        for bar in barsX.keys():
            barX = bar[0][0]
            barY1, barY2 = min(bar[0][1], bar[1][1]), max(bar[0][1], bar[1][1])

            for n in nodesToCheck:
                nodeX, nodeY = px[n], py[n]
                if dir > 0 and barX > nodeX:
                    if barY1 <= nodeY <= barY2:
                        dist = barX - nodeX
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)
                elif dir < 0 and barX < nodeX:
                    if barY1 <= nodeY <= barY2:
                        dist = nodeX - barX
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)

        for n in nodesToCheck:
            nodeX, nodeY = px[n], py[n]
            for ox, oy in static.values():
                if oy == nodeY:
                    if dir > 0 and ox > nodeX:
                        dist = ox - nodeX
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)
                    elif dir < 0 and ox < nodeX:
                        dist = nodeX - ox
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)
    else:
        # Moving vertically - check horizontal bars and nodes
        for bar in barsY.keys():
            barY = bar[0][1]
            barX1, barX2 = min(bar[0][0], bar[1][0]), max(bar[0][0], bar[1][0])

            for n in nodesToCheck:
                nodeX, nodeY = px[n], py[n]
                if dir > 0 and barY > nodeY:
                    if barX1 <= nodeX <= barX2:
                        dist = barY - nodeY
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)
                elif dir < 0 and barY < nodeY:
                    if barX1 <= nodeX <= barX2:
                        dist = nodeY - barY
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)

        for n in nodesToCheck:
            nodeX, nodeY = px[n], py[n]
            for ox, oy in static.values():
                if ox == nodeX:
                    if dir > 0 and oy > nodeY:
                        dist = oy - nodeY
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)
                    elif dir < 0 and oy < nodeY:
                        dist = nodeY - oy
                        units = int(dist // spacing) - 1
                        minUnits = min(minUnits, units)

    moveUnits = max(0, minUnits)

    if moveUnits > 0:
        for n in nodesToCheck:
            if isHorizPrimary:
                positions[n] = (
                    positions[n][0] + dir * moveUnits * spacing,
                    positions[n][1],
                )
            else:
                positions[n] = (
                    positions[n][0],
                    positions[n][1] + dir * moveUnits * spacing,
                )
        print(f"{node} (+{len(ancestralBlock)}) -> ({moveUnits}) {target}")
        return True
    return False


def checkLineIntersection(p1, p2, p3, p4):
    """
    Check if line segment p1-p2 intersects with line segment p3-p4
    Returns True if they intersect
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Using the formula for line segment intersection
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:  # Lines are parallel
        return False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Intersection exists if 0 <= t <= 1 and 0 <= u <= 1
    return 0 <= t <= 1 and 0 <= u <= 1


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
            if (
                node not in ancestors
                and node in descendants
                and descendants[node] not in bars[b]
            ):
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
            if (
                node not in ancestors
                and node in descendants
                and descendants[node] not in bars[b]
            ):
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
        if p not in descendants:  # Skip if p has no descendants
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
        if p not in descendants:  # Skip if p has no descendants
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
    with open(rf"{os.getcwd()}/out/{file}-H.svg", "r") as f:
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
        # split node key into base id and numeric suffix (handles multi-digit suffixes)
        if "-" in p:
            key = p.rsplit("-", 1)[0]
            suffix = p.rsplit("-", 1)[1]
        else:
            key = p
            suffix = "0"
        # Only draw connection lines if the node has descendants
        if p in descendants:
            xc, yc = initialPositions[descendants[p]]
            rx = {"M": int(size / 6), "F": int(size / 2)}[people[key].gender]
            tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' rx='{rx}' width='{
                size}' height='{size}' class='tree{' primary' if p0 == key else ''}' id='{p}' />"
            tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' width='{
                size}' height='{size}' style='{gen(people[key].generation)}'/>"
            if people[key].name.last:
                try:
                    is_dup = int(suffix) != 0
                except Exception:
                    is_dup = False
                names += f"<text class='name{' duplicate' if is_dup else ''}'><tspan x='{x}' y='{y}' dy='-2.5pt'>{people[key].name.first or ''}</tspan><tspan x='{x}' y='{y}' dy='7.5pt'>{people[key].name.last or ''}</tspan></text>"
            else:
                names += f"<text class='name{' duplicate' if int(p[-1]) != 0 else ''}'><tspan x='{
                    x}' y='{y}'>{people[key].name.first}</tspan></text>"
            lines += f"<path d='M {x},{y} L {xc},{yc}' class='line'/>"
        else:
            # Draw just the node without connections for leaf nodes
            rx = {"M": int(size / 6), "F": int(size / 2)}[people[key].gender]
            tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' rx='{rx}' width='{
                size}' height='{size}' class='tree{' primary' if p0 == key else ''}' id='{p}' />"
            tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' width='{
                size}' height='{size}' style='{gen(people[key].generation)}'/>"
            if people[key].name.last:
                try:
                    is_dup = int(suffix) != 0
                except Exception:
                    is_dup = False
                names += f"<text class='name{' duplicate' if is_dup else ''}'><tspan x='{x}' y='{y}' dy='-2.5pt'>{people[key].name.first or ''}</tspan><tspan x='{x}' y='{y}' dy='7.5pt'>{people[key].name.last or ''}</tspan></text>"
            else:
                names += f"<text class='name{' duplicate' if int(p[-1]) != 0 else ''}'><tspan x='{
                    x}' y='{y}'>{people[key].name.first}</tspan></text>"
    root = svg.find("svg")
    if root is None:
        raise RuntimeError(f"No <svg> element found in template {file}-H.svg")
    root.clear()
    for fragment in (styles, lines, tree, names):
        frag_soup = bs4.BeautifulSoup(fragment, "html.parser")
        root.extend(frag_soup)
    root.attrs.update(
        {
            "viewBox": f"{minx - size / 2} {miny - size / 2} {maxx - minx + size} {maxy - miny + size}"
        }
    )
    with open(rf"{os.getcwd()}/out/{file}-H.svg", "w") as f:
        f.write(svg.prettify())


def getInitialPositionsH(s, N, p0) -> Tuple[Dict, Dict, Dict]:
    initialPositions = {}
    descendants = {}
    ancestors = {}
    done = set()
    current = {p0}

    personPositionMap = {}

    while current:
        for c in list(current):
            personDescent = descent(c, p0)
            for d in personDescent:
                x, y = positionH(d, N, s=s + 15)
                pos = (x, y)

                personPosKey = (c, pos)
                if personPosKey in personPositionMap:
                    key = personPositionMap[personPosKey]
                else:
                    existingKeysForPerson = [
                        k for k in initialPositions.keys() if k.startswith(c + "-")
                    ]
                    suffix = str(len(existingKeysForPerson))
                    key = f"{c}-{suffix}"

                    personPositionMap[personPosKey] = key
                    initialPositions[key] = (x, y)

                done.add(c)

                if len(d) > 1:
                    parentPosition = positionH(d[1:], N)
                    parentPerson = d[1]

                    parentPosKey = (parentPerson, parentPosition)
                    if parentPosKey in personPositionMap:
                        parentKey = personPositionMap[parentPosKey]
                    else:
                        existingParentKeys = [
                            k
                            for k in initialPositions.keys()
                            if k.startswith(parentPerson + "-")
                        ]
                        parentSuffix = str(len(existingParentKeys))
                        parentKey = f"{parentPerson}-{parentSuffix}"

                        personPositionMap[parentPosKey] = parentKey
                        initialPositions[parentKey] = parentPosition

                    if key not in descendants:
                        descendants[key] = parentKey
                        ancestors.setdefault(parentKey, set()).add(key)

            if people[c].father in people:
                current.add(people[c].father)
            if people[c].mother in people:
                current.add(people[c].mother)
            current.discard(c)
    return ancestors, descendants, initialPositions


def positionH(d: list[str], N: int, s=90 + 15) -> Tuple[int, int]:
    x, y = 0, 0
    if len(d) == 1:
        return int(x), int(y)
    for i, j in enumerate(d[::-1][1:]):
        if not (i + 1) % 2:
            y += (
                (-1 if people[j].gender == "M" else 1)
                * s
                * 2 ** (N - (people[j].generation or 0) // 2)
            )
        else:
            x += (
                (-1 if people[j].gender == "M" else 1)
                * s
                * 2 ** (N - (people[j].generation or 0) // 2 - 1)
            )
    return int(x), int(y)


def getHArea(initialPositions: Dict, descendants: Dict) -> Dict[str, int]:
    minx = min([p[0] for p in initialPositions.values()])
    maxx = max([p[0] for p in initialPositions.values()])
    miny = min([p[1] for p in initialPositions.values()])
    maxy = max([p[1] for p in initialPositions.values()])
    edgeTotal = 0
    for p in initialPositions:
        if p not in descendants:
            continue
        edge = getChildDist(descendants, initialPositions, p)
        edgeTotal += edge
    return {"maxx": maxx, "maxy": maxy, "minx": minx, "miny": miny, "edge": edgeTotal}


def getChildDist(descendants: Dict, initialPositions: Dict, p: str) -> int:
    if p not in descendants:
        return 0  # Return 0 if node has no descendants
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
    anc: Dict, des: Dict, pos: Dict
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[str]]:
    """
    Create bars representing nodes that share the same x-coordinate and cannot be crossed by horizontal movement
    O(n log n) version - group by coordinate in single pass
    """
    # Group nodes by x coordinate in a single pass - O(n)
    axis: Dict = {}
    for node, (x, y) in pos.items():
        if x not in axis:
            axis[x] = []
        axis[x].append((node, y))

    # Sort each group by y coordinate - O(n log n) total
    for x in axis:
        axis[x].sort(key=lambda item: item[1])

    bars: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[str]] = {}
    for x in axis:
        done = set()
        y1, y2, k = None, None, []
        nodes_at_x = axis[x]
        for i, (u, y) in enumerate(nodes_at_x):
            if u in done:
                continue
            k.append(u)
            y1 = y if y1 is None else min(y1, y)
            y2 = y if y2 is None else max(y2, y)
            if i + 1 == len(nodes_at_x):
                bars.update({((x, y1), (x, y2)): k})
                y1, y2, k = None, None, []
                break
            v = nodes_at_x[i + 1][0]
            if not v == des.get(u) and v not in anc.get(u, set()):
                bars.update({((x, y1), (x, y2)): k})
                y1, y2, k = None, None, []
            done.add(u)
    return bars


def getYBars(
    anc: Dict, des: Dict, pos: Dict
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[str]]:
    """
    Create bars representing nodes that share the same y-coordinate and cannot be crossed by vertical movement
    O(n log n) version - group by coordinate in single pass
    """
    # Group nodes by y coordinate in a single pass - O(n)
    axis: Dict = {}
    for node, (x, y) in pos.items():
        if y not in axis:
            axis[y] = []
        axis[y].append((node, x))

    # Sort each group by x coordinate - O(n log n) total
    for y in axis:
        axis[y].sort(key=lambda item: item[1])

    bars: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[str]] = {}
    for y in axis:
        done = set()
        x1, x2, k = None, None, []
        nodes_at_y = axis[y]
        for i, (u, x) in enumerate(nodes_at_y):
            if u in done:
                continue
            k.append(u)
            x1 = x if x1 is None else min(x1, x)
            x2 = x if x2 is None else max(x2, x)
            if i + 1 == len(nodes_at_y):
                bars.update({((x1, y), (x2, y)): k})
                x1, x2, k = None, None, []
                break
            v = nodes_at_y[i + 1][0]
            if not v == des.get(u) and v not in anc.get(u, set()):
                bars.update({((x1, y), (x2, y)): k})
                x1, x2, k = None, None, []
            done.add(u)
    return bars


def getEdgeBars(ancestors, descendants, positions):
    """Create bars representing parent-child connections that cannot be crossed"""
    bars = []

    # Add connections from ancestors to their children
    for parent, children in ancestors.items():
        if parent in positions:
            parentX, parentY = positions[parent]
            for child in children:
                if child in positions:
                    childX, childY = positions[child]
                    # Store as a tuple of start and end points
                    bars.append(((parentX, parentY), (childX, childY)))

    # Add connections from nodes to their descendants
    for node, desc in descendants.items():
        if node in positions and desc in positions:
            nodeX, nodeY = positions[node]
            descX, descY = positions[desc]
            # Store as a tuple of start and end points
            bars.append(((nodeX, nodeY), (descX, descY)))

    return bars


def generateLineTree(file, root):
    currentYear: int = datetime.datetime.now().year + 10
    done, unknownb, unknownd = getApproxVitals(root)

    numGenerations = len(generations)
    ancestors, descendants, initialPositions = getInitialPositionsLine(
        pt2px(10), numGenerations, root
    )

    # Use unit-based movement with collision checking instead of visibility-based compaction
    positions = initialPositions.copy()
    spacing = pt2px(15)  # Use appropriate spacing

    # Continue moving ancestors toward children until no more movements possible
    moved = True
    max_iterations = 100  # Prevent infinite loops
    iterations = 0

    while moved and iterations < max_iterations:
        moved = moveAncestorsTowardsChildren(positions, ancestors, descendants, spacing)
        iterations += 1

    for p in done:
        done[p]["y"] = positions[p]
    minY = min(done[p]["y"] for p in done)
    for p in done:
        done[p]["y"] = done[p]["y"] - minY

    drawLineChart(currentYear, done, file, unknownb, unknownd)
    return positions


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
        if people[p].mother in done:
            m = people[p].mother
            begats += f"<path d='M {done[p]['b'] * w /
                              widx:.3f},{addPt(p, 'y', -5):.3f} " f"V {addPt(m, 'y', 5):.3f}' " f"class='{'unbegat' if p in unknownb |
                          unknownd else 'begat'}' id='{m}-{p}'/>"
            begats += f"<circle cx='{done[p]['b'] * w /
                                     widx:.3f}' cy='{addPt(m, 'y', -5):.3f}' r='3' />"
        if people[p].father in done:
            f = people[p].father
            begats += f"<path d='M {done[p]['b'] * w /
                              widx:.3f},{addPt(p, 'y', 5):.3f} " f"V {addPt(f, 'y', -5):.3f}' " f"class='{'unbegat' if p in unknownb |
                          unknownd else 'begat'}' id='{f}-{p}'/>"
            begats += f"<circle cx='{done[p]['b'] * w /
                                     widx:.3f}' cy='{addPt(f, 'y'):.3f}' r='3' />"
        if people[p].mother in done:
            begats += f"<circle cx='{done[p]['b'] * w /
                                     widx:.3f}' cy='{addPt(p, 'y'):.3f}' r='3' />"
    for p in done:
        if p in unknownb and p not in unknownd:
            p_class = "unknownb"
            if getState(p, "death"):
                flags += f"<image x='{
                        (done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " f"height='9pt' href='../flags/{
                        getState(p, 'death').lower()}.png'/>"
        elif p not in unknownb and p in unknownd:
            p_class = "unknownd"
            if getState(p, "birth"):
                flags += f"<image x='{
                        (done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " f"height='9pt' href='../flags/{getState(p, 'birth').lower(
                    )}.png' style='transform: translateX(-15.72pt)'/>"
        elif p in unknownb and p in unknownd:
            p_class = "unknownbd"
        else:
            p_class = "known"
            if getState(p, "death"):
                flags += f"<image x='{
                        (done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " f"height='9pt' href='../flags/{
                        getState(p, 'death').lower()}.png'/>"
            if getState(p, "birth"):
                flags += f"<image x='{
                        (done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " f"height='9pt' href='../flags/{getState(p, 'birth').lower(
                    )}.png' style='transform: translateX(-15.72pt)'/>"
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
        for s in people[p].spouse:
            if not people[p].marriage[s].getYear():
                continue
            flags += (
                f"<image x='{people[p].marriage[s].getYear() * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' "
                f"height='9pt' href='../flags/{getState(p, 'marriage')[s].lower(
                )}.png' style='transform: translateX(-7.36pt)'/>"
            )
    for p in done:
        names += f"<text class='name {
                'nameunknown' if p in unknownb | unknownd else ''}'>" f"<tspan dx='{pt2px(2):.3f}' dy='{pt2px(1):.3f}' x={
                done[p]['b'] * w / widx:.3f} y={done[p]['y']:.3f}>" f"{people[p].name.first} {
                people[p].name.last}</tspan></text>"
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
                if key != keyc:
                    descendants.update({key: keyc})
                    ancestors.setdefault(keyc, set())
                    ancestors[keyc].add(key)
            if (father := people[c].father) and father in people:
                current.add(father)
            if (mother := people[c].mother) and mother in people:
                current.add(mother)
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


def getAncestralBlock(
    node: str, ancestors: Dict[str, Set[str]], max_size: int = 7
) -> Set[str]:
    """Get the ancestral block for a node (node + its ancestors up to max_size)"""
    block = {node}
    to_process = [node]

    while to_process and len(block) < max_size:
        current = to_process.pop(0)
        if current in ancestors:
            for ancestor in ancestors[current]:
                if ancestor not in block and len(block) < max_size:
                    block.add(ancestor)
                    to_process.append(ancestor)

    return block


def moveAncestorsTowardsChildren(
    positions: Dict[str, float],
    ancestors: Dict[str, Set[str]],
    descendants: Dict[str, str],
    spacing: float,
) -> bool:
    """Move ancestors toward their children with collision checking, moving ancestral blocks together"""
    moved = False

    # Process nodes from bottom to top (children before parents)
    sorted_nodes = sorted(positions.keys(), key=lambda x: positions[x], reverse=True)

    for node in sorted_nodes:
        if node not in descendants:  # Skip if node doesn't have a child
            continue

        child = descendants[node]
        if child not in positions:
            continue

        node_pos = positions[node]
        child_pos = positions[child]

        # Only move if the parent is above the child and they're too far apart
        if node_pos < child_pos and (child_pos - node_pos) > spacing:
            # Get the ancestral block for this node (node + its ancestors up to a limit)
            ancestral_block = getAncestralBlock(node, ancestors, max_size=7)

            # Calculate target position for the node
            target_pos = child_pos - spacing
            if target_pos <= node_pos:
                continue

            # Check for collisions with other nodes
            collision = False
            for other_node, other_pos in positions.items():
                if other_node in ancestral_block or other_node == child:
                    continue
                # Check if any node in the block would cross this node
                for block_node in ancestral_block:
                    block_pos = positions[block_node]
                    block_target = block_pos + (target_pos - node_pos)
                    if (block_pos < other_pos < block_target) or (
                        block_target < other_pos < block_pos
                    ):
                        collision = True
                        break
                if collision:
                    break

            if not collision:
                # Move the entire ancestral block
                delta = target_pos - node_pos
                for block_node in ancestral_block:
                    positions[block_node] = positions[block_node] + delta
                moved = True

    return moved


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


def positionLine(descentList: list, numGenerations: int) -> int:
    y: int = 0
    if len(descentList) == 1:
        return 0
    for _, j in enumerate(descentList[::-1][1:]):
        yy = round(
            (1 if people[j].gender == "F" else -1)
            * 2 ** (numGenerations - (people[j].generation or 0))
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
                lines += f"<path id='{d['id']}-{s}' d='M {people[d['id']].marriage[s].getYear():.3f} {y(i) + toPt(5):.3f} V {y(sidx) + toPt(5):.3f}' />"
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
            if (
                marriageYear := people[people[p].father]
                .marriage[people[p].mother]
                .getYear()
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
            "b": people[p].birth.getYear() or getApproxBirth(dict(), p),
            "d": people[p].death.getYear(),
        }

    descent = [essentials(p)]
    spouses = getSpouse(p)
    for spouse in spouses:
        if spouse:
            if people[p].gender == "F":
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
    approxVitals = getApproxVitals(root)[0]
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
                                + (1 if people[p].gender == "M" else 0) * d,
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
        paths += f'<path id="{r}" class="life {people[r].gender}" d="M {radialChart[r]["r1"]} 0 H {
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
