import os
from typing import Dict, Literal, Set, Union
import warnings

from models import Marriage, Person, Vitals
from utils import (
    getAntonym,
    getFullName,
    getParent,
    getTitle,
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
    if not p1 or p1 not in people:
        return ""
    parentLine = getLineage(p1, parent)
    line: str = f"\\namelink{{{p1}}}{{{people[p1].name.first}}}"
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
    history = person.history
    childrenDetails = getChildrenDetails(person, p0)
    spouses = sorted(
        filter(lambda s: s != "", person.spouse),
        key=lambda x: person.marriage[x].getYear() or 3000,
    )
    burialDetails: list[str] = [getBurialDetails(person)] + [
        getBurialDetails(people[s]) for s in spouses if s
    ]
    burialDetails = list(filter(lambda b: b != "", burialDetails))
    if len(set(burialDetails)) != 1:
        for i, (b, q) in enumerate(zip(burialDetails, [p] + spouses)):
            burialDetails[i] = f"{b} ({people[q].name.first})"
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


def getSources(person: Person) -> str:
    if not person.sources:
        return ""
    allSources = person.sources
    for s in person.spouse:
        if not s:
            continue
        allSources.update(people.get(s, Person).sources)
    sources = [f"\\item\\fullcite[{allSources[s]}]{{{s}}}" for s in sorted(allSources)]
    return buildSentence("\\begin{source}", *sources, "\\end{source}")


def getBurialDetails(person: Person) -> str:
    if person.buried.cemetery:
        if person.buried.plusCode:
            return (
                f"\\buried \\href{{{'http://plus.codes/' + person.buried.plusCode}}}"
                f"{{{person.buried.cemetery}}}"
            )
        return f"\\buried {person.buried.cemetery}"
    return ""


def getChildrenDetails(person: Person, p0: str) -> str:
    childrens = []
    parentDetails = ""
    for s in person.getFecundSpouses():
        if s:
            parentDetails = getParentDetails(person, s) + "\n"
        children = []
        for c in sorted(
            person.marriage[s].children,
            key=lambda y: people[y].birth.getYear() or 3000,
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


def getChildDetails(person: Person, c: str, p0: str) -> str:
    mainLine = getMainLine(person, c, p0)
    title = getTitle(people[c])
    antonym = getAntonym(people[c])
    birth = childBirth(c)
    marriage = childMarriage(c, mainLine, p0)
    return (
        f"\\childlist{mainLine}{{{c if mainLine else ''}}}"
        f"{{{buildSentence(title, joinComma(
            people[c].name.shortname, antonym))}}}"
        f"{{{buildParagraph(birth, marriage)}}}"
    )


def childMarriage(c: str, mainLine: str, p0: str) -> str:
    if people[c].gender == "M" or not mainLine:
        return ""
    spouses = people[c].spouse
    if type(spouses) == str:
        spouses = [spouses]
    for cs in spouses:
        if not people.get(cs, Person).generation:
            continue
        return f"{people[c].getPronoun()} married {getShortNamelink(cs, p0)}"


def childBirth(c: str) -> str:
    return f"born {people[c].birth.date}" if people[c].birth.date else ""


def getMainLine(person: Person, c: str, p0: str) -> str:
    return "[+]" if inFullTree(c, p0) else ""


def getParentDetails(person: Person, s: str) -> str:
    if not s:
        return f"{person.name.shortname}\\children"
    if s not in people:
        return f"{person.name.shortname} and {s}\\children"
    return f"{person.name.shortname} and {people[s].name.shortname}\\children"


def generateSpouse(person: Person, p0: str):
    spouse = person.spouse
    if type(spouse) == str:
        spouse = [spouse]
    spouseDetail = []
    sortedSpouses = sorted(
        filter(lambda s: s != "" and not person.marriage[s].adulterous, spouse),
        key=lambda x: person.marriage[x].getYear() or 3000,
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
        history = people[s].history
        spouseMarriageHistory = generateSpouseMarriageHistory(person, s)

        spouseDetail += [
            buildSentence(
                person.getPronoun(), "married", nSpouse, spouseName, marriage
            ),
            buildSentence(people[s].getPronoun() if vitals else None, vitals),
            history,
            spouseMarriageHistory,
        ]
    return spouseDetail


def generateSpouseMarriageHistory(person: Person, s: str) -> str:
    spouse = people[s]
    spouseDetail = []
    sortedSpouses = sorted(
        filter(lambda s: s != "" and not person.marriage[s].adulterous, spouse.spouse),
        key=lambda x: spouse.marriage[x].getYear() or 3000,
    )
    if len(sortedSpouses) <= 1:
        return ""
    for s in sortedSpouses:
        spouseName = getSpouseName(s, person.id, includePatriline=False)
        nSpouse = getSpouseNumber(s, sortedSpouses)
        spouseDetail.append(
            buildSentence(nSpouse, spouseName),
        )
    if not spouseDetail:
        return ""
    if len(spouseDetail) == 1:
        joined = spouseDetail[0]
    elif len(spouseDetail) == 2:
        joined = " and ".join(spouseDetail)
    else:
        joined = ", ".join(spouseDetail[:-1]) + " and " + spouseDetail[-1]
    return f"{spouse.getPronoun()} married {joined}"


def getSpouseParents(s: str, p0: str) -> str:
    spouseFather = people[s].father
    spouseMother = people[s].mother
    spouseFatherName = None
    spouseMotherName = None
    if spouseFather in people:
        spouseFatherName = getSpouseName(spouseFather, p0, includePatriline=False)
    if spouseMother in people:
        spouseMotherName = getSpouseName(spouseMother, p0, includePatriline=False)
    return " and ".join(filter(None, [spouseFatherName, spouseMotherName]))


def getShortNamelink(p: str, p0: str) -> str:
    if inFullTree(p, p0):
        return f"\\namelink{{{p}}}{{{people[p].name.shortname}}}"
    return f"{people[p].name.shortname}"


def getShortNamelinkBold(p: str, p0: str) -> str:
    if inFullTree(p, p0):
        return f"\\namelinkbold{{{p}}}{{{people[p].name.shortname}}}"
    return f"{people[p].name.shortname}"


def getSpouseName(s: str, p0: str, includePatriline: bool = True) -> str:
    patriline = printLineage(s, "father")
    shortName = people[s].name.shortname
    if inFullTree(s, p0):
        if not people[s].father and not people[s].mother:
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


def buildParagraphs(*paragraphs: iter) -> str:
    return "\n\n".join(filter(None, paragraphs))


def buildParagraph(*sentences: iter) -> str:
    paragraph = ". ".join(filter(None, sentences))
    return paragraph + ("." if not paragraph.endswith("quote}") else "")


def buildSentence(*phrases: iter) -> str:
    return " ".join(filter(None, phrases))


def getAccolades(person: Person) -> str:
    accolades = []
    for a in ["army", "mason"]:
        if getattr(person, a, None):
            accolades.append(a)
    if blazon := getattr(person, "blazon", set()):
        for b in blazon:
            accolades.append(rf"includegraphics[height=\fontcharht\font`l]{{{b}}}")
    return r"\,".join(rf"\{a}" for a in accolades)


def combineVitals(birth: str, death: str, parents: str = "") -> str:
    if birth:
        birth = f"was born {birth}"
    if parents:
        birth += f" to {parents}" if birth else f"was born to {parents}"
    if death:
        death = f"died {death}"
    vitals = "; ".join(filter(None, [birth, death]))
    return vitals


def combineDatePlace(person: Person, vital: Literal["birth", "death"]) -> str:
    if vital not in ["birth", "death"]:
        raise KeyError(f"{vital} is not a vital statistic")
    vitalDate = getattr(getattr(person, vital, Vitals), "date", "")
    vitalPlace = getattr(getattr(person, vital, Vitals), "place", "")
    date = ""
    place = ""
    if vitalDate:
        full = isDateFull(vitalDate)
        date = f"{'on' if full else 'in'} {vitalDate}"
    if vitalPlace:
        place = f"in {vitalPlace}"
    return joinComma(date, place)


def combineMarriageDatePlace(person: Person, s: str) -> str:
    vitalDate = getattr(person.marriage.get(s, Marriage), "date")
    vitalPlace = getattr(person.marriage.get(s, Marriage), "place")
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


def getNameIndex(person: Person) -> str:
    firstName = person.name.first
    middleName = person.name.middle
    lastName = person.name.last
    nameIndex = rf"\index{{{lastName or ''}!{
        joinName(firstName, middleName)}}}"
    return nameIndex


def getAncestorTag(person: Person) -> str:
    if not person.father:
        return "[p]"
    return ""


def generateTex(familyName: str, p0: str):
    importFamily(familyName, p0)
    with open(rf"{os.getcwd()}/tex/{familyName}_generated.tex", "w") as f:
        writeTitle(f, familyName)
        writeGenerations(f, p0)
    print(f"{len(set.union(*generations))} total ancestors")


def writeGenerations(f, p0: str):
    for g in generations:
        writeGeneration(f, p0, g)


def writeGeneration(f, p0: str, g: Set):
    if p0 in g and people[p0].gender == "F":
        return
    f.write(f"\\generationgroup\n\n")
    for p in sorted(list(g), key=lambda y: people[y].birth.getYear() or 3000):
        if people[p].gender == "F" and people[p].spouse:
            continue
        f.write(f"{printIndividualEntry(p, p0)}\n\n")


def writeTitle(f, familyName: str):
    f.write(f"\\chapter*{{{familyName.capitalize()}}}\n\n")
