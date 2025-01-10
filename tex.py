import os
from typing import Dict, Literal, Set, Union
import warnings

from utils import (
    getAntonym,
    getFullName,
    getMarriageYear,
    getParent,
    getTitle,
    getVitalYear,
    importFamily,
    inFullTree,
    isDateFull,
    joinName,
)

from config import people, generations


def getLineage(p: str, parent: Literal["father", "mother"]) -> str:
    """
    Get the lineage of a person
    :param p: the person to get the lineage of
    :param parent: the parent to get the lineage of
    :return: the lineage of the person in TeX format
    """
    if parent not in ["father", "mother"]:
        raise KeyError(f"{parent} is not a proper parent")
    p1 = getParent(p, parent)
    p2 = getParent(p, {"father": "mother"}.get(parent, "father"))
    if not p1 or p1 not in people:
        return ""
    parentLine = getLineage(p1, parent)
    line: str = (
        f"\\namelink{{{p1}}}{{{
            people[p1].get('name', {}).get('first')}}}"
    )
    line += ", " if parentLine else ""
    line += parentLine
    return line


def printIndividualEntry(p: str, p0: str) -> str:
    """
    Print an individual TeX entry for a person
    :param p: the person to print the entry for
    :param p0: the root person
    :return: the individual entry for the person
    """
    # child_check(p)
    person = people[p]
    ancestor = getAncestorTag(person)
    name = getFullName(person)
    nameIndex = getNameIndex(person)
    title = getTitle(person)
    antonym = getAntonym(person)
    patriline = printLineage(p, "father")
    birth = combineDatePlace(person, "birth")
    death = combineDatePlace(person, "death")
    vitals = combineVitals(birth, death)
    accolades = getAccolades(person)
    spouseDetails = generateSpouse(person, p0)
    history = person.get("history", None)
    childrenDetails = getChildrenDetails(person, p0)
    spouses = sorted(
        filter(lambda s: s != "", person.get("spouse", [])),
        key=lambda x: getMarriageYear(person, x),
    )
    burialDetails: list[str] = [getBurialDetails(person)] + [
        getBurialDetails(people[s]) for s in spouses
    ]
    burialDetails = list(filter(lambda b: b != "", burialDetails))
    if len(set(burialDetails)) != 1:
        for i, (b, q) in enumerate(zip(burialDetails, [p] + spouses)):
            burialDetails[i] = (
                f"{
                    b} ({people[q].get('name', {}).get('first')})"
            )
    burialDetails = list(set(burialDetails))
    sources = getSources(person)

    return buildParagraphs(
        buildParagraph(
            buildSentence(
                rf"\individual{ancestor}{{{p}}}{{{buildSentence(
                    title, joinComma(name, antonym))}{nameIndex}}}",
                accolades,
                patriline,
                vitals,
            ),
            *spouseDetails,
            history,
        ),
        childrenDetails,
        buildParagraphs(*burialDetails),
        sources,
    )


def getSources(person: Dict) -> str:
    if not person.get("sources"):
        return ""
    allSources = person.get("sources")
    for s in person.get("spouse"):
        if not s:
            continue
        allSources |= people[s].get("sources", set())
    sources = [f"\\item{{{s}}}" for s in sorted(allSources)]
    return buildSentence("\\begin{source}", *sources, "\\end{source}")


def getBurialDetails(person: Dict) -> str:
    if person.get("buried"):
        return (
            f"\\buried \\href{{{'http://plus.codes/' +
                                person.get('buried', {}).get('plusCode', '')}}}"
            f"{{{person.get('buried', {}).get('cemetery')}}}"
        )
    return ""


def getChildrenDetails(person: Dict, p0: str) -> str:
    childrens = []
    for s in person.get("children"):
        if not person.get("children")[s]:
            continue
        parentDetails = getParentDetails(person, s) + "\n"
        children = []
        for c in sorted(
            person.get("children")[s], key=lambda y: getVitalYear(y, "birth") or 3000
        ):
            if c not in people:
                warnings.warn(f"{c} doesn't have an entry", Warning)
                continue
            childDetail = getChildDetails(person, c, p0)
            children += [childDetail]
        if children:
            children.insert(0, parentDetails)
            childrens.append("\n".join(children))
    return "\n".join(childrens)


def getChildDetails(person: Dict, c: str, p0: str) -> str:
    mainLine = getMainLine(person, c)
    title = getTitle(people[c])
    antonym = getAntonym(people[c])
    birth = childBirth(c)
    marriage = childMarriage(c, mainLine, p0)
    return (
        f"\\childlist{mainLine}{{{c if mainLine else ''}}}"
        f"{{{buildSentence(title, joinComma(
            people[c]['shortname'], antonym))}}}"
        f"{{{buildParagraph(birth, marriage)}}}"
    )


def childMarriage(c: str, mainLine: str, p0: str) -> str:
    if people[c]["gender"] == "M" or not mainLine:
        return ""
    spouses = people[c].get("spouse", "")
    if type(spouses) == str:
        spouses = [spouses]
    for cs in spouses:
        if not people.get(cs, dict()).get("generation"):
            continue
        return f"{getPronoun(people[c])} married {getShortNamelink(cs, p0)}"


def childBirth(c: str) -> str:
    if people[c].get("birth", {}).get("date"):
        return f"born {people[c].get('birth', {}).get('date')}"
    return ""


def getMainLine(person: Dict, c: str) -> str:
    if c in person.get("child", set()):
        return "[+]"
    return ""


def getParentDetails(person: Dict, s: str) -> str:
    if not s:
        return f"{person['shortname']}\\children"
    if s not in people:
        return f"{person['shortname']} and {s}\\children"
    return f"{person['shortname']} and {people[s]['shortname']}\\children"


def generateSpouse(person: Dict, p0: str):
    spouse: list = person.get("spouse", [])
    if type(spouse) == str:
        spouse: list = [spouse]
    spouseDetail = []
    sortedSpouses = sorted(
        filter(lambda s: s != "", spouse), key=lambda x: getMarriageYear(person, x)
    )
    for s in sortedSpouses:
        if not s:
            continue
        spouseName = getSpouseName(s, p0)
        nSpouse = getSpouseNumber(s, sortedSpouses)
        marriage = combineMarriageDatePlace(person, s)
        birth = combineDatePlace(people[s], "birth")
        death = combineDatePlace(people[s], "death")
        vitals = combineVitals(birth, death, parents=getSpouseParents(s, p0))
        history = people[s].get("history")

        spouseDetail += [
            buildSentence(getPronoun(person), "married", nSpouse, spouseName, marriage),
            buildSentence(getPronoun(people[s]) if vitals else None, vitals),
            history,
        ]
    return spouseDetail


def getSpouseParents(s: str, p0: str) -> str:
    spouseFather = people[s].get("father")
    spouseMother = people[s].get("mother")
    spouseFatherName = ""
    spouseMotherName = ""
    if spouseFather in people:
        spouseFatherName = getSpouseName(spouseFather, p0, includePatriline=False)
    if spouseMother in people:
        spouseMotherName = getSpouseName(spouseMother, p0, includePatriline=False)
    return " and ".join(filter(None, [spouseFatherName, spouseMotherName]))


def getShortNamelink(p: str, p0: str) -> str:
    if inFullTree(p, p0):
        return f"\\namelink{{{p}}}{{{people[p]['shortname']}}}"
    return f"{people[p]['shortname']}"


def getShortNamelinkBold(p: str, p0: str) -> str:
    if inFullTree(p, p0):
        return f"\\namelinkbold{{{p}}}{{{people[p]['shortname']}}}"
    return f"{people[p]['shortname']}"


def getSpouseName(s: str, p0: str, includePatriline: bool = True) -> str:
    patriline = printLineage(s, "father")
    shortName = people[s]["shortname"]
    if inFullTree(s, p0):
        if not people[s].get("father") and not people[s].get("mother"):
            return f"\\textbf{{{shortName}}}{getNameIndex(people[s])}"
        if includePatriline:
            return buildSentence(getShortNamelinkBold(s, p0), f"{patriline}")
        return f"{getShortNamelinkBold(s, p0)}"
    return f"\\textbf{{{shortName}}}"


def getSpouseNumber(s: str, spouse: iter) -> Union[int, str]:
    nSpouse = spouse.index(s) + 1
    if len(spouse) > 1:
        return f"({nSpouse})"
    return ""


def getPronoun(person: Dict) -> str:
    return {"M": "He"}.get(person["gender"], "She")


def buildParagraphs(*paragraphs: iter) -> str:
    return "\n\n".join(filter(None, paragraphs))


def buildParagraph(*sentences: iter) -> str:
    paragraph = ". ".join(filter(None, sentences))
    return paragraph + ("." if not paragraph.endswith("quote}") else "")


def buildSentence(*phrases: iter) -> str:
    return " ".join(filter(None, phrases))


def getAccolades(person: Dict) -> str:
    accolades = []
    for a in ["army", "mason"]:
        if person.get(a):
            accolades.append(a)
    return "".join(f"\\{a}" for a in accolades)


def combineVitals(birth: str, death: str, parents: str = "") -> str:
    if birth:
        birth = f"was born {birth}"
    if parents:
        birth += f" to {parents}" if birth else f"was born to {parents}"
    if death:
        death = f"died {death}"
    vitals = "; ".join(filter(None, [birth, death]))
    return vitals


def combineDatePlace(person: Dict, vital: str) -> str:
    if vital not in ["birth", "death", "marriage"]:
        raise KeyError(f"{vital} is not a vital statistic")
    vitalDate: str = person.get(vital, {}).get("date")
    vitalPlace: str = person.get(vital, {}).get("place")
    date: str = ""
    place: str = ""
    if vitalDate:
        full = isDateFull(vitalDate)
        date = f"{'on' if full else 'in'} {vitalDate}"
    if vitalPlace:
        place = f"in {vitalPlace}"
    return joinComma(date, place)


def combineMarriageDatePlace(person: Dict, s: str) -> str:
    vitalDate = person.get("marriage", {}).get(s, {}).get("date")
    vitalPlace = person.get("marriage", {}).get(s, {}).get("place")
    date: str = ""
    place: str = ""
    if vitalDate:
        full = isDateFull(vitalDate)
        date = f"{'on' if full else 'in'} {vitalDate}"
    if vitalPlace:
        place = f"in {vitalPlace}"
    return joinComma(date, place)


def joinComma(*phrases) -> str:
    return ", ".join(filter(None, phrases))


def printLineage(p, parent):
    patriline = getLineage(p, parent)
    if not patriline:
        return ""
    return f"({patriline})"


def getNameIndex(person: Dict) -> str:
    firstName = person.get("name", {}).get("first")
    middleName = person.get("name", {}).get("middle")
    lastName = person.get("name", {}).get("last")
    nameIndex = rf"\index{{{lastName or ''}!{
        joinName(firstName, middleName)}}}"
    return nameIndex


def getAncestorTag(person: Dict) -> str:
    if not person.get("father"):
        return "[p]"
    return ""


def generateTex(familyName: str, p0: str):
    importFamily(familyName, p0)
    with open(rf"{os.getcwd()}\out\{familyName}_generated.tex", "w") as f:
        writeTitle(f, familyName)
        writeGenerations(f, p0)
    print(f"{len(set.union(*generations))} total ancestors")


def writeGenerations(f, p0: str):
    for g in generations:
        writeGeneration(f, p0, g)


def writeGeneration(f, p0: str, g: Set):
    if p0 in g and people[p0]["gender"] == "F":
        return
    f.write(f"\\generationgroup\n\n")
    for p in sorted(list(g), key=lambda y: getVitalYear(y, "birth") or 3000):
        if people[p]["gender"] == "F" and people[p].get("spouse", []):
            continue
        f.write(f"{printIndividualEntry(p, p0)}\n\n")


def writeTitle(f, familyName: str):
    f.write(f"\\chapter*{{{familyName}}}\n\n")
