import datetime
import json
import os
from typing import Dict, List, Literal, Set, Union
import warnings
from functools import lru_cache
from collections import deque

import pluscodes
from requests_cache import Optional

from config import people, generations
from models import Cousin, Marriage, Marriages, Person, Vitals


def isDateFull(d: Union[str, int]) -> bool:
    """
    Check if a date is full (meaning, it has a day, month, and year)
    :param d: the date to check
    :return: True if the date is full, False otherwise
    """
    d = str(d).split(" ")
    return len(d) == 3


def loadRawData(familyName: str) -> Dict:
    """
    Load the raw data from a JSON file
    :param familyName: the name of the JSON file
    :return: the raw data
    """
    with open(rf"{os.getcwd()}/data/{familyName}.tree.json", "r") as f:
        rawData = json.load(f)
    if not runLinter(rawData):
        raise KeyError
    return rawData


def importFamily(familyName: str, p0: str):
    """
    Import a family tree from a JSON file
    :param familyName: the name of the JSON file
    :param p0: the root person
    :return: None
    """
    people.clear()
    rawData = loadRawData(familyName)
    # initial setup
    for p in rawData:
        people[p["id"]] = Person(**{k: v for k, v in p.items()})
    for p in list(people):
        mar = getattr(people[p], "marriage", {}) or {}
        people[p].marriage = Marriages({s: Marriage(**m) for s, m in mar.items()})
    # process children
    for p in list(people):
        for s in people[p].getFecundSpouses():
            for c in list(people[p].marriage[s].children):
                if not c:
                    people[p].marriage[s].children.remove(c)
                    continue
                if "|" in c:
                    newPerson = generateFromShorthand(c, p)
                    newId = newPerson.id
                    people[newId] = newPerson
                    people[p].marriage[s].children.remove(c)
                    people[p].marriage[s].children.add(newId)
                    c = newId
                if people[p].gender == "M":
                    people[c].father = p
                    people[c].mother = s
                else:
                    people[c].father = s
                    people[c].mother = p
    # reciprocate data to spouses
    for p in list(people):
        for s in list(people[p].spouse):
            if s and "|" in s:
                newPerson = generateFromShorthand(s)
                newId = newPerson.id
                people[newId] = newPerson
                people[p].marriage.update({newId: people[p].marriage[s]})
                people[p].marriage.pop(s)
                people[p].spouse.add(newId)
                people[p].spouse.remove(s)
                people[newId].marriage = Marriages({p: people[p].marriage[newId]})
                people[newId].spouse |= set([p])
                for child in people[p].marriage[newId].children:
                    people[child].father, people[child].mother = (
                        (p, newId) if people[p].gender == "M" else (newId, p)
                    )
            if s in people:
                people[s].spouse |= set([p])
                people[s].marriage.update({p: people[p].marriage[s]})
    # Clear ancestor cache (used by inFullTree/getAncestors) because `people` mutated
    try:
        getAncestors.cache_clear()
    except Exception:
        pass
    updateGenerationGroups(p0)


def runLinter(rawData: Dict) -> bool:
    """
    Lint the JSON file for duplicates
    :param rawData: raw data from the JSON file
    :return: True if no duplicates are found, False otherwise
    """
    ids: Set[str] = set()
    linted = True
    for person in rawData:
        if person["id"] in ids:
            warnings.warn(f"{person['id']} is a duplicate ID", Warning)
            linted = False
            continue
        ids.add(person["id"])
    return linted


def inFullTree(p: str, p0: str) -> bool:
    """
    Check if a person is related to the root person
    :param p: the person to check
    :param p0: the root person
    :return: True if the person is related, False otherwise
    """
    return p in getAncestors(p0) or p == p0


def getParent(p: str, parent: Literal["father", "mother"]) -> Union[str, None]:
    """
    Get the parent of a person
    :param p: the person to get the parent of
    :param parent: one of "father", "mother"
    :return: the parent of the person
    """
    if parent not in ["father", "mother"]:
        raise KeyError(f"{parent} is not a proper parent")
    return getattr(people.get(p, Person), parent, None)


def getTitle(person: Person) -> str:
    return person.name.title


def getAntonym(person: Person) -> str:
    return person.name.antonym


def descent(p: str, p0: str) -> list[list[str]]:
    d = [[p]]
    found_paths: list[list[str]] = []

    while d:
        current_path = d.pop(0)
        current = current_path[-1]

        if current == p0:
            found_paths.append(current_path)
            continue

        for spouse in people[current].getFecundSpouses():
            for child in people[current].marriage[spouse].children:
                if child:
                    new_path = current_path + [child]
                    d.append(new_path)

    return sorted(found_paths, key=lambda m: lenSpearLine(m[::-1]), reverse=True)


def lenSpearLine(m: list) -> int:
    i = int(people[m[0]].gender == "M")
    c = 0
    for j in m[i:]:
        c += int(people[j].gender == "M")
    return c


def updateGenerationGroups(p0):
    generations.clear()
    setGenerations(p0)
    maxG = max(people[p].generation or 0 for p in people)
    for g in range(maxG + 1):
        generations.insert(0, set(filter(lambda p: isInGeneration(g, p), people)))


def setGenerations(p0):
    # Reset generations
    for p in people:
        people[p].generation = None

    if p0 not in people:
        warnings.warn(f"root {p0} not found; skipping generation assignment", Warning)
        return

    # BFS up the parent links from p0 to assign generation (distance) to each ancestor.
    q = deque()
    people[p0].generation = 0
    q.append(p0)

    while q:
        cur = q.popleft()
        cur_gen = people[cur].generation or 0
        for parent in (people[cur].father, people[cur].mother):
            if parent and parent in people:
                parent_gen = cur_gen + 1
                if (
                    people[parent].generation is None
                    or parent_gen > people[parent].generation
                ):
                    people[parent].generation = parent_gen
                    q.append(parent)


def isInGeneration(g: int, p: str) -> bool:
    return people[p].generation == g


@lru_cache(maxsize=None)
def getAncestors(p):
    ancestors = {p}
    i = 0
    while i != len(ancestors):
        i = len(ancestors)
        for q in list(ancestors):
            if people[q].father in people:
                ancestors.add(people[q].father)
            if people[q].mother in people:
                ancestors.add(people[q].mother)
    ancestors.discard(p)
    return ancestors


def childCheck(p):
    gtg = True
    for s in people[p].getFecundSpouses():
        for c in people[p].marriage[s].children:
            if people.get(c, Person).generation >= people[p].generation:
                warnings.warn(f"generation issue: {p} and {c}", Warning)
                gtg = False
    return updateGenerationGroups


def follow(p0: str, g=None, lost=False):
    i = 0
    endOfLine = dict()
    for p in sorted(people, key=lambda p: people[p].birth.getYear() or 3000):
        if lost and people[p].lost:
            continue
        if g and people[p].generation != g:
            continue
        if people[p].gender == "F" and people[p].spouse and people[p].generation:
            if type(people[p].spouse) == set:
                spouseNote = ", ".join([people[s].note for s in people[p].spouse if s])
            else:
                spouseNote = people[people[p].spouse].note
            if not people[p].father:
                endOfLine.update(
                    {
                        p: {
                            "spouse": people[p].spouse,
                            "birthyear": people[p].birth.getYear(),
                            "note": people[p].note,
                            "spouseNote": spouseNote,
                        }
                    }
                )
                i += 1
            elif not inFullTree(people[p].father, p0):
                endOfLine.update(
                    {
                        p: {
                            "spouse": people[p].spouse,
                            "birthyear": people[p].birth.getYear(),
                            "note": people[p].note,
                            "spouseNote": spouseNote,
                        }
                    }
                )
                i += 1
        if (
            people[p].gender == "M"
            and people[p].generation
            and not inFullTree(people[p].father, p0)
        ):
            endOfLine.update(
                {
                    p: {
                        "birthyear": people[p].birth.getYear(),
                        "note": people[p].note,
                    }
                }
            )
        i += 1
    for line in sorted(endOfLine, key=lambda p: endOfLine[p].get("birthyear") or 0):
        print(line, endOfLine[line])
    print(f"{i} threads to pull")
    return


def unsourced(g=None):
    i = 0
    for g in generations[::-1][2:]:
        for p in g:
            if people[p].gender == "M" and not people[p].sources:
                print(
                    p,
                    people[p].name.shortname,
                    people[p].birth.getYear(),
                    people[p].note,
                )
                i += 1
    print(f"{i} sources to get")
    return


def generationCount(children=False):
    if children:
        for g in generations:
            N = len(
                set.union(
                    *(
                        [
                            set.union(
                                *(
                                    [
                                        people[p].marriage[s].children
                                        for s in people[p].marriage.keys()
                                    ]
                                    if people[p].marriage
                                    else [set()]
                                )
                            )
                            for p in g
                        ]
                    )
                )
            )
            print(f"{generations.index(g) + 1}\t{N}\t{len(g)}")
    else:
        for g in generations:
            print(f"{generations.index(g) + 1}\t{len(g)}")


def getApproxVitals(root):
    currentYear: int = datetime.datetime.now().year
    current: Set[str] = {root}
    done: Dict[str, Dict[str, int | None]] = dict()
    while current:
        for p in list(current):
            b = people[p].birth.getYear()
            d = people[p].death.getYear()
            y = 0
            done.update({p: {"b": b, "d": d, "y": y}})
            if people[p].father in people:
                current.add(people[p].father or "")
            if people[p].mother and people[p].mother in people:
                current.add(people[p].mother or "")
            current.discard(p)

    unknownb: Set[str] = set()
    unknownd: Set[str] = set()
    for p in done:
        if not people[p].death.getYear():
            if (
                people[p].birth.getYear()
                and currentYear - (people[p].birth.getYear() or currentYear) < 100
            ):
                done[p]["d"] = currentYear
                continue
            unknownd.add(p)
            done[p]["d"] = getApproxDeath(done, p)
        if not people[p].birth.getYear():
            unknownb.add(p)
            done[p]["b"] = getApproxBirth(done, p)
    return done, unknownb, unknownd


def getChildBirthYears(done: dict, p: str) -> int | None:
    allChildren = getAllChildren(p)
    childBirthYears: int | None = list(
        filter(None, [getApproxBirth(done, c) for c in allChildren])
    ) or [None]
    return childBirthYears


def getMarriageDates(p: str):
    marriageDates = []
    if people[p].gender == "M":
        spouses = people[p].spouse
        marriageDates = [people[p].marriage[s].getYear() for s in spouses]
    elif people[p].spouse:
        spouse: str = (
            list(people[p].spouse)[0]
            if type(people[p].spouse) == set
            else people[p].spouse
        )
        marriageDates = [people[p].marriage[spouse].getYear()]
    marriageDates = max(
        list(filter(lambda x: x != 0, filter(None, marriageDates))) or [None]
    )
    return marriageDates


def getApproxDeath(done: dict, p: str) -> int | None:
    if getattr(getattr(people[p], "death", Vitals), "date", False):
        return people[p].death.getYear()
    lastChildBirth = max(getChildBirthYears(done, p))
    marriageDates = getMarriageDates(p)
    return max(list(filter(None, [lastChildBirth, marriageDates])) or [None])


def getApproxBirth(done: dict, p: str) -> int | None:
    if getattr(getattr(people[p], "birth", Vitals), "date", False):
        return people[p].birth.getYear()

    firstChildBirth = min(getChildBirthYears(done, p))
    if firstChildBirth:
        firstChildBirth -= 20
    fatherDeath: int | None = done.get(people[p].mother, dict()).get("d")
    motherDeath: int | None = done.get(people[p].father, dict()).get("d")
    marriageDates = getMarriageDates(p)
    if marriageDates:
        marriageDates -= 20
    return min(
        list(filter(None, [fatherDeath, motherDeath, firstChildBirth, marriageDates]))
        or [None]
    )


def getAllChildren(p: str) -> Set[str]:
    allChildren = {i for i in people if p in {people[i].mother, people[i].father}}
    return allChildren


def birthdays(month, p0):
    for p in people:
        if inFullTree(p, p0) and month in str(people[p].get("birth", {}).get("date")):
            announce(p, p0)
            print(" ")


def announce(p, p0):
    for d in descent(p, p0):
        a = f"On {people[p].birth.date}, {
            people[p].name.shortname} was born"
        if people[p].birth.place:
            a += f" in {people[p].birth.place}.{' ' +
                                                people[p].history or ''}\n"
        for n, i in enumerate(d[:-1]):
            if people[i].title:
                a += f"{people[i].title} "
            if people[i].spouse:
                if people[i].gender == "M":
                    for s in people[i].spouse:
                        if d[n + 1] in people[i]["children"][s]:
                            break
                else:
                    s = (
                        people[i].spouse[0]
                        if type(people[i].spouse) == set
                        else people[i].spouse
                    )
                if s:
                    a += f"{people[i].name.shortname} married {
                        people[s].name.shortname} and begat {people[d[n + 1]].name.shortname}.\n"
                else:
                    a += f"{people[i].name.shortname} begat {
                        people[d[n + 1]].name.shortname}.\n"
            else:
                a += f"{people[i].name.shortname} begat {
                    people[d[n + 1]].name.shortname}.\n"
        a += f"{people[p].name.first} was my {'great-' * (len(d) - 2)}grand{'father' if people[p].gender == 'M' else 'mother'}."
        if people[p].buried:
            a += f"\n{'He' if people[p].gender == 'M' else 'She'} is buried in {people[p].buried.cemetery}."
        print(a)
    return


def getLivingAncestors(p: str) -> Set[str]:
    b0 = people[p].birth.getYear()
    d0 = people[p].death.getYear()
    alive = set()
    if not b0 and not d0:
        raise ValueError(f"{p} doesnt have vital statistics")
    year = datetime.date.today().year
    for a in getAncestors(p):
        b = people[a].birth.getYear()
        d = people[a].death.getYear()
        if not d and year - b < 100:
            d = year
        if b0 and d and d > b0:
            alive.add(a)
    return alive


def getState(
    p: dict, time: Literal["marriage", "birth", "death"]
) -> Union[None, dict, str]:
    if time == "marriage":
        state = {
            m: (
                getattr(
                    getattr(people[p], "marriage", {}).get(m, Marriage), "place", ","
                )
                or ","
            )
            .split(",")[-1]
            .strip()
            for m in getattr(people[p], "marriage", {"": ","})
        }
        return state
    if not getattr(getattr(people[p], time, Vitals), "place", False):
        return None
    state = getattr(getattr(people[p], time, Vitals), "place", ",").split(",")
    return state[-1].strip()


def ancestors_by_age():
    byAge = [
        p
        for p in set.union(*generations)
        if people[p].get("birth", {}).get("date") and people[p]["deathdate"]
    ]
    for p in sorted(
        byAge, key=lambda p: people[p].death.getYear() - people[p].birth.getYear()
    ):
        print(
            people[p]["name"]["shortname"],
            people[p].death.getYear() - people[p].birth.getYear(),
        )


def generateID(name: dict) -> dict:
    """
    Generate an ID for a person
    :param name: the name of the person
    :return: an ID"""
    nameID = (name.get("first", "") + name.get("last", "")).replace(" ", "").lower()
    nameID = nameID.replace(".", "").replace("'", "")
    name.update({"id": generateIDn(nameID)})
    return name


def generateIDn(nameID: str) -> str:
    """
    Generate a unique ID for a person
    :param nameID: the name of the person
    :return: a unique ID"""
    i = 1
    while True:
        if (nameIDi := nameID + str(i) if i > 1 else nameID) not in people:
            return nameIDi
        i += 1


def getSpouse(p: str) -> List[str]:
    if len(people[p].spouse) == 0:
        return ""
    if type(people[p].spouse) == str:
        return [people[p].spouse]
    return people[p].spouse


def getChildren(p: str, spouse: str):
    children = list(
        people[p].get("marriage", {}).get(spouse, Marriage).get("children", set())
        | people.get(spouse, Marriage)
        .get("marriage", {})
        .get(p, Marriage)
        .get("children", set())
    )
    children.sort(
        key=lambda c: (people[c].birth.getYear() is None, people[c].birth.getYear())
    )
    return children


def dms(old):
    direction = {"N": 1, "S": -1, "E": 1, "W": -1}
    new = old.replace("°", " ").replace("'", " ").replace('"', " ")
    new = new.split()
    new_dir = new.pop()
    new.extend([0, 0, 0])
    return (int(new[0]) + int(new[1]) / 60.0 + int(new[2]) / 3600.0) * direction[
        new_dir
    ]


def getCoordinates(link):
    coords = pluscodes.decode(link).center()
    return f"{coords.lat},{coords.lon}"


def printGraves():
    for p in people:
        if "plusCode" in people[p].get("buried", {}):
            print(f'{getFullName(people[p])},"{getCoordinates(
                    people[p]["buried"]["plusCode"])}"')


def begats(p: str, p0: str) -> str | list[str]:
    dd = descent(p, p0)
    begat = []
    for i, d in enumerate(dd):
        begat.append("")
        for j, person in enumerate(d):
            if j == 0:
                begat[
                    i
                ] += f'{people[person].get("name", {}).get("first")} {people[person].get("name", {}).get("last")} begat {people[d[j + 1]].get("name", {}).get("first")}'
            elif j < len(d) - 1:
                if people[person].get("name", {}).get("last") == people[d[j + 1]].get(
                    "name", {}
                ).get("last"):
                    begat[
                        i
                    ] += f', {people[person].get("name", {}).get("first")} begat {people[d[j + 1]].get("name", {}).get("first")}'
                else:
                    begat[
                        i
                    ] += f', {people[person].get("name", {}).get("first")} begat {people[d[j + 1]].get("name", {}).get("first")} {people[d[j + 1]].get("name", {}).get("last")}'
        begat[i] += "."
    return begat


def getFullName(person: Person) -> str:
    firstName = person.name.first
    middleName = person.name.middle
    lastName = person.name.last
    name = joinName(firstName, middleName, lastName)
    return name


def joinName(*name) -> str:
    joinedName = " ".join(filter(None, name))
    return joinedName


def parseDate(date: Union[str, int]) -> Union[str, int]:
    if type(date) == int:
        return date
    date = date.split(" ")
    if len(date) > 1:
        return joinName(*date)
    return int(date[0])


def generateFromShorthand(c: str, p: str = "") -> Person:
    """
    Generate a person from a shorthand
    :param c: the shorthand
    :param p: the parent
    :return: the person
    """
    gender, first, last, birth = (c.split("|") + [""] * 4)[:4]
    if not first.startswith("\\AE"):
        first = first.capitalize()
    if isRoman(first.split(" ")[-1]):
        first = first.split(" ")
        first = (
            " ".join([f.capitalize() for f in first[:-1]])
            + " "
            + first[-1].strip().upper()
        )
    if last:
        if "'" in last:
            last = last.split("'")
            last = (
                ("'".join(last[:-1]) + "'" + last[-1].capitalize())
                if len(last) > 1
                else last[0].capitalize()
            )
        else:
            last = last.split(" ")
            last = (
                (" ".join(last[:-1]) + " " + last[-1].capitalize())
                if len(last) > 1
                else last[0].capitalize()
            )
    elif p:
        last = people[p].name.last

    newPerson = Person(
        **{
            "id": generateIDn(f"{first.lower()}{last.lower()}"),
            "name": {"first": first, "last": last},
            "gender": gender,
        }
    )
    if birth:
        newPerson.birth = Vitals(date=birth)

    return newPerson


def cousins(p1, p2) -> list[Cousin]:
    """
    Returns cousin relationships (first, second, etc.) between p1 and p2 based on their common ancestors.
    If there are many pathways between two people, this will return a list of all cousin relationships.

    :param p1: person 1
    :param p2: person 2
    :return: a list of cousin relationships (e.g. "first cousin", "second cousin once removed", etc.)
    """
    p1Ancestors = getAncestors(p1) | {p1}
    p2Ancestors = getAncestors(p2) | {p2}
    commonAncestors = p1Ancestors & p2Ancestors
    if not commonAncestors:
        return []
    dist = {}
    for c in commonAncestors:
        d1 = descent(c, p1)
        d2 = descent(c, p2)
        dist[c] = {p1: [len(i) - 1 for i in d1], p2: [len(i) - 1 for i in d2]}
    minAncestor = min(dist, key=lambda c: min(*dist[c][p1], *dist[c][p2]))
    cousins: list[Cousin] = []
    for g in dist[minAncestor][p1]:
        for h in dist[minAncestor][p2]:
            degree = min(g, h)
            removed = abs(g - h)
            cousins.append(Cousin(degree, removed))
    return cousins


def isRoman(s: str) -> bool:
    romanNumerals = set("IVXLCDM")
    return all(c in romanNumerals for c in s.upper())
