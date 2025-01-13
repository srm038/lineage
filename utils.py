import datetime
import json
import os
from typing import Dict, List, Literal, Set, Union
import warnings

import pluscodes
from requests_cache import Optional

from config import people, generations
from types_node import Marriage, Person


def isDateFull(d: Union[str, int]) -> bool:
    """
    Check if a date is full (meaning, it has a day, month, and year)
    :param d: the date to check
    :return: True if the date is full, False otherwise
    """
    d = str(d)
    d = d.split(" ")
    return len(d) == 3


def loadRawData(familyName: str) -> Dict:
    """
    Load the raw data from a JSON file
    :param familyName: the name of the JSON file
    :return: the raw data
    """
    with open(rf"{os.getcwd()}\data\{familyName}.tree.json", "r") as f:
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
        pid = p["id"]
        people.update({pid: {k: v for k, v in p.items() if k != "id"}})
        person = people[pid]
        person["gender"] = person["gender"].upper()
        person.setdefault("marriage", {})
        person["spouse"] = list()
        for s in person["marriage"]:
            if s:
                person["spouse"].append(s)
            if "children" in person["marriage"][s]:
                person["marriage"][s]["children"] = set(
                    person["marriage"][s]["children"]
                )
        for i in ["child", "sources"]:
            person.setdefault(i, set())
            person[i] = set(person[i])
        for i in ["first", "middle", "last"]:
            if i not in person:
                continue
            person[i] = person[i].strip()
            person[i] = person[i].replace(":", '\\"')
        if "shortname" not in person["name"]:
            person["name"]["shortname"] = joinName(
                person.get("name", {}).get("first"), person.get("name", {}).get("last")
            )
    # process children
    for p in list(people):
        for s in people[p].get("marriage", {}):
            for c in people[p]["marriage"][s].get("children", set()):
                if not c:
                    people[p]["marriage"][s]["children"].remove(c)
                    continue
                if c not in people:
                    newPerson = generateFromShorthand(c, p)
                    newId = newPerson["id"]
                    people[newId] = newPerson
                    people[p]["marriage"][s]["children"].remove(c)
                    people[p]["marriage"][s]["children"].add(newId)
                    c = newId
                parent = "father" if people[p]["gender"] == "M" else "mother"
                otherParent = "mother" if parent == "father" else "father"
                people[c][parent] = p
                if s and s in people:
                    people[c][otherParent] = s
    # reciprocate data to spouses
    for p in list(people):
        if people[p]["gender"] == "M":
            for s in list(people[p].get("marriage", {})):
                if s and s not in people:
                    newPerson = generateFromShorthand(s)
                    newId = newPerson["id"]
                    people[newId] = newPerson
                    people[p]["marriage"].update({newId: people[p]["marriage"][s]})
                    people[p]["marriage"].pop(s)
                    people[p]["spouse"].append(newId)
                    people[p]["spouse"].remove(s)
                    people[newId]["marriage"] = {p: people[p]["marriage"][newId]}
                    people[newId]["spouse"] = p
                    people[newId]["child"] = {
                        c
                        for c in people[p]["child"]
                        if c in people[newId]["marriage"][p].get("children", set())
                    }
                if s in people:
                    people[s]["spouse"] = p
                    people[s].setdefault("marriage", {})
                    people[s]["marriage"].update({p: people[p]["marriage"][s]})
                    people[s]["child"] = {
                        c
                        for c in people[p]["child"]
                        if c in people[s]["marriage"][p].get("children", set())
                    }
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
    return people.get(p, {}).get(parent)


def getTitle(person: Person) -> str:
    return person.get("title", "")


def getAntonym(person: Person) -> str:
    return person.get("antonym", "")


def descent(p: str, p0: str) -> list:
    d = [[p]]
    while d[0][-1] != p0:
        children = list(people[d[0][-1]]["child"])
        if not len(children):
            print(people[d[0][-1]])
        if len(children) > 1:
            d.append(d[-1] + descent(children[1], p0)[0])
        d[0].append(children[0])

    return sorted(d, key=lambda m: lenSpearLine(m[::-1]), reverse=True)


def lenSpearLine(m: list) -> int:
    i = int(people[m[0]]["gender"] == "M")
    c = 0
    for j in m[i:]:
        c += int(people[j]["gender"] == "M")
    return c


def updateGenerationGroups(p0):
    generations.clear()
    setGenerations(p0)
    maxG = max(people[p].get("generation", 0) for p in people)
    for g in range(maxG + 1):
        generations.insert(0, set(filter(lambda p: isInGeneration(g, p), people)))


def setGenerations(p0):
    for p in filter(lambda p: inFullTree(p, p0), people):
        people[p]["generation"] = max(len(d) - 1 for d in descent(p, p0))


def isInGeneration(g: int, p: str) -> bool:
    return "generation" in people[p] and people[p]["generation"] == g


def getAncestors(p):
    ancestors = {p}
    i = 0
    while i != len(ancestors):
        i = len(ancestors)
        for q in list(ancestors):
            if people[q].get("father") in people:
                ancestors.add(people[q].get("father"))
            if people[q].get("mother") in people:
                ancestors.add(people[q].get("mother"))
    ancestors.discard(p)
    return ancestors


def childCheck(p):
    for s in people[p].get("marriage", {}):
        for c in people[p]["marriage"][s].get("children", set()):
            if people.get(c, Person).get("generation", 100) >= people[p].get(
                "generation", 0
            ):
                warnings.warn(f"generation issue: {p} and {c}", Warning)


def getVitalYear(p: str, vital: Literal["birth", "death"]) -> Optional[int]:
    if vital not in ["birth", "death"]:
        raise KeyError(f"{vital} is not a vital statistic")
    date: Union[int, str] = people.get(p, {}).get(vital, {}).get("date", 0)
    if not date:
        return None
    if type(date) == int:
        return date
    date: list = date.split(" ")
    return int(date[-1])


def getMarriageYear(person: Person, s: str) -> int:
    if not s:
        return 0
    date: Union[int, str] = person.get("marriage", {}).get(s, Marriage).get("date", 0)
    if not date:
        return 0
    if type(date) == int:
        return date
    date: list = date.split(" ")
    return int(date[-1])


def follow(p0: str, g=None, lost=False):
    i = 0
    endOfLine = dict()
    for p in sorted(people, key=lambda p: getVitalYear(p, "birth") or 3000):
        if lost and people[p].get("lost"):
            continue
        if g and people[p].get("generation") != g:
            continue
        if (
            people[p]["gender"] == "F"
            and people[p].get("spouse")
            and people[p].get("generation")
        ):
            if type(people[p].get("spouse")) == list:
                spouseNote = ", ".join(
                    [people[s].get("note") for s in people[p]["spouse"] if s]
                )
            else:
                spouseNote = people[people[p]["spouse"]].get("note")
            if not people[p].get("father"):
                endOfLine.update(
                    {
                        p: {
                            "spouse": people[p]["spouse"],
                            "birthyear": getVitalYear(p, "birth"),
                            "note": people[p].get("note"),
                            "spouseNote": spouseNote,
                        }
                    }
                )
                i += 1
            elif not inFullTree(people[p].get("father"), p0):
                endOfLine.update(
                    {
                        p: {
                            "spouse": people[p]["spouse"],
                            "birthyear": getVitalYear(p, "birth"),
                            "note": people[p].get("note"),
                            "spouseNote": spouseNote,
                        }
                    }
                )
                i += 1
        if (
            people[p]["gender"] == "M"
            and people[p].get("generation")
            and not inFullTree(people[p].get("father"), p0)
        ):
            endOfLine.update(
                {
                    p: {
                        "birthyear": getVitalYear(p, "birth"),
                        "note": people[p].get("note"),
                    }
                }
            )
        i += 1
    for line in sorted(endOfLine, key=lambda p: endOfLine[p].get("birthyear") or 0):
        print(line, endOfLine[line])
    print(f"{i} threads to pull")


def unsourced(g=None):
    i = 0
    for g in generations[::-1][2:]:
        for p in g:
            if people[p]["gender"] == "M" and not people[p]["sources"]:
                print(
                    p,
                    people[p]["name"]["shortname"],
                    getVitalYear(p, "birth"),
                    people[p]["note"],
                )
                i += 1
    print(f"{i} sources to get")
    # return


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
                                        people[p]["marriage"]
                                        .get(s, Marriage)
                                        .get("children", set())
                                        for s in people[p].get("marriage", {})
                                    ]
                                    if people[p].get("marriage", {})
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
    done: Dict[str, Dict[str, int]] = dict()
    while current:
        for p in list(current):
            b = getVitalYear(p, "birth")
            d = getVitalYear(p, "death")
            y = 0
            done.update({p: {"b": b, "d": d, "y": y}})
            if people[p].get("father") in people:
                current.add(people[p]["father"])
            if people[p].get("mother") in people:
                current.add(people[p]["mother"])
            current.discard(p)

    unknownb: Set[str] = set()
    unknownd: Set[str] = set()
    for p in done:
        if not getVitalYear(p, "death"):
            if (
                getVitalYear(p, "birth")
                and currentYear - getVitalYear(p, "birth") < 100
            ):
                done[p]["d"] = currentYear
                continue
            unknownd.add(p)
            done[p]["d"] = getApproxDeath(done, p)
        if not getVitalYear(p, "birth"):
            unknownb.add(p)
            done[p]["b"] = getApproxBirth(done, p)
    return done, unknownb, unknownd


def getApproxDeath(done: dict, p: str) -> int:
    if people[p].get("death", {}).get("date"):
        return getVitalYear(p, "death")
    allChildren = getAllChildren(p)
    lastChildBirth: int = max(
        list(filter(None, [getApproxBirth(done, c) for c in allChildren])) or [None]
    )
    if people[p]["gender"] == "M":
        spouses = people[p].get("spouse", [])
        marriageDates: List[Optional[int]] = [
            getMarriageYear(people[p], s) for s in spouses
        ]
    else:
        spouse = (
            people[p].get("spouse")[0]
            if type(people[p].get("spouse")) == list
            else people[p].get("spouse")
        )
        marriageDates: List[Optional[int]] = [getMarriageYear(people[p], spouse)]
    marriageDates: Optional[int] = max(
        list(filter(lambda x: x != 0, marriageDates)) or [None]
    )
    return max(list(filter(None, [lastChildBirth, marriageDates])) or [None])


def getApproxBirth(done: dict, p: str) -> int:
    if people[p].get("birth", {}).get("date"):
        return getVitalYear(p, "birth")
    allChildren = getAllChildren(p)
    firstChildBirth: Optional[int] = min(
        list(filter(None, [getApproxBirth(done, c) for c in allChildren])) or [None]
    )
    if firstChildBirth:
        firstChildBirth -= 20
    fatherDeath: Optional[int] = done.get(people[p].get("mother"), dict()).get("d")
    motherDeath: Optional[int] = done.get(people[p].get("father"), dict()).get("d")
    if people[p]["gender"] == "M":
        marriageDates: List[Optional[int]] = [
            getMarriageYear(people[p], s) for s in people[p].get("spouse", [])
        ]
    else:
        marriageDates: List[Optional[int]] = [
            getMarriageYear(people[p], people[p].get("spouse"))
        ]
    marriageDates: Optional[int] = min(
        list(filter(lambda x: x != 0, marriageDates)) or [None]
    )
    if marriageDates:
        marriageDates -= 20
    return min(
        list(filter(None, [fatherDeath, motherDeath, firstChildBirth, marriageDates]))
        or [None]
    )


def getAllChildren(p: str) -> Set[str]:
    allChildren = {
        i for i in people if p in {people[i].get("mother"), people[i].get("father")}
    }
    return allChildren


def birthdays(month, p0):
    for p in people:
        if inFullTree(p, p0) and month in str(people[p].get("birth", {}).get("date")):
            announce(p, p0)
            print(" ")


def announce(p, p0):
    for d in descent(p, p0):
        a = f"On {people[p].get('birth', {}).get('date')}, {
            people[p]['name']['shortname']} was born"
        if people[p]["birthplace"]:
            a += f" in {people[p]['birthplace']}.{' ' + people[p]
                                                  ['history'] if people[p]['history'] else ''}\n"
        for n, i in enumerate(d[:-1]):
            if people[i]["title"]:
                a += f"{people[i]['title']} "
            if people[i]["spouse"]:
                if people[i]["gender"] == "M":
                    for s in people[i]["spouse"]:
                        if d[n + 1] in people[i]["children"][s]:
                            break
                else:
                    s = (
                        people[i]["spouse"][0]
                        if type(people[i]["spouse"]) == list
                        else people[i]["spouse"]
                    )
                if s:
                    a += f"{people[i]['name']['shortname']} married {people[s]['name']
                                                             ['shortname']} and begat {people[d[n + 1]]['name']['shortname']}.\n"
                else:
                    a += f"{people[i]['name']['shortname']
                            } begat {people[d[n + 1]]['shortname']}.\n"
            else:
                a += f"{people[i]['name']['shortname']} begat {
                    people[d[n + 1]]['name']['shortname']}.\n"
        a += f"{people[p].get('name', {}).get('first')} was my {'great-' * (
            len(d) - 2)}grand{'father' if people[p]['gender'] == 'M' else 'mother'}."
        if people[p]["buried"]:
            a += f"\n{'He' if people[p]['gender'] == 'M' else 'She'} is buried in {
                people[p]['buried']['cemetery']}."
        print(a)


def getLivingAncestors(p: str) -> Set[str]:
    b0 = getVitalYear(p, "birth")
    d0 = getVitalYear(p, "death")
    alive = set()
    if not b0 and not d0:
        raise ValueError(f"{p} doesnt have vital statistics")
    year = datetime.date.today().year
    for a in getAncestors(p):
        b = getVitalYear(a, "birth")
        d = getVitalYear(a, "death")
        if not d and year - b < 100:
            d = year
        if b0 and d and d > b0:
            alive.add(a)
    return alive


def getState(p: dict, time: str) -> Union[None, dict, str]:
    if time == "marriage":
        state = {
            m: people[p]
            .get("marriage", {})
            .get(m, {})
            .get("place", ",")
            .split(",")[-1]
            .strip()
            for m in people[p].get("marriage", {"": ","})
        }
        return state
    if not people[p].get(time, {}).get("place"):
        return None
    state = people[p].get(f"{time}place", ",").split(",")
    return state[-1].strip()


def ancestors_by_age():
    byAge = [
        p
        for p in set.union(*generations)
        if people[p].get("birth", {}).get("date") and people[p]["deathdate"]
    ]
    for p in sorted(
        byAge, key=lambda p: getVitalYear(p, "death") - getVitalYear(p, "birth")
    ):
        print(
            people[p]["name"]["shortname"],
            getVitalYear(p, "death") - getVitalYear(p, "birth"),
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
    if len(people[p]["spouse"]) == 0:
        return ""
    if type(people[p]["spouse"]) == str:
        return [people[p]["spouse"]]
    return people[p]["spouse"]


def getChildren(p: str, spouse: str):
    children = list(
        people[p].get("marriage", {}).get(spouse, Marriage).get("children", set())
        | people.get(spouse, Marriage)
        .get("marriage", {})
        .get(p, Marriage)
        .get("children", set())
    )
    children.sort(
        key=lambda c: (getVitalYear(c, "birth") is None, getVitalYear(c, "birth"))
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
            print(
                f'{getFullName(people[p])},"{getCoordinates(
                    people[p]["buried"]["plusCode"])}"'
            )


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
    firstName = person.get("name", {}).get("first")
    middleName = person.get("name", {}).get("middle")
    lastName = person.get("name", {}).get("last")
    name = joinName(firstName, middleName, lastName)
    return name


def joinName(*name: iter) -> str:
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
    first = first.capitalize()
    if last:
        last = last.capitalize()
    elif p:
        last = people[p].get("name", {}).get("last", "")

    newPerson: Person = {
        "id": generateIDn(f"{first.lower()}{last.lower()}"),
        "name": {
            "first": first,
            "shortname": joinName(first, last),
        },
        "gender": gender,
    }
    if last:
        newPerson["name"]["last"] = last
    if birth:
        newPerson["birth"] = {"date": parseDate(birth)}

    return newPerson
