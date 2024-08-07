import copy
import json
import math
import os
import datetime
import bs4
from typing import *
import warnings
import re

people = dict()
generations: List[Set] = list()


def isDateFull(d: Union[str, int]) -> bool:
    d = str(d)
    d = d.split(' ')
    return len(d) == 3


def importFamily(familyName: str, p0: str):
    people.clear()
    with open(fr"{os.getcwd()}\{familyName}.json", 'r') as f:
        rawData = json.load(f)
    if not runLinter(rawData):
        raise KeyError
    for p in rawData:
        pid = p['id']
        people.update({pid: {k: v for k, v in p.items() if k != 'id'}})
        person = people[pid]
        person['gender'] = person['gender'].upper()
        for i in ['marriagedate', 'marriageplace', 'children']:
            person.setdefault(i, dict())
        for i in person['children']:
            person['children'][i] = set(person['children'][i])
        for i in ['child', 'sources']:
            person.setdefault(i, set())
            person[i] = set(person[i])
        for i in ['first', 'middle', 'last']:
            if i not in person:
                continue
            person[i] = person[i].strip()
            person[i] = person[i].replace(':', "\\\"")
        if 'shortname' not in person:
            person['shortname'] = joinName(person.get('first'), person.get('last'))
        for i in ['marriagedate', 'marriageplace', 'children']:
            spouses = list(person.get(i, set()))
            person.setdefault('spouse', [])
            person['spouse'].extend(spouses)
        person['spouse'] = list(set(person['spouse']))
    for p in people:
        if people[p]['gender'] == 'M':
            for s in people[p].get('spouse', set()):
                if s in people:
                    # people[s].setdefault('spouse', list()).extend([p])
                    people[s]['spouse'] = p
                    if s in people[p]['marriagedate']:
                        people[s]['marriagedate'].update({p: people[p]['marriagedate'][s]})
                    if s in people[p]['marriageplace']:
                        people[s]['marriageplace'].update({p: people[p]['marriageplace'][s]})
            for s in people[p].get('children', dict()):
                if not s or s not in people:
                    continue
                for c in people[p].get('child', set()):
                    if c in people[p]['children'][s]:
                        people[s]['child'].add(c)
    for p in list(people):
        if people[p]['gender'] == 'M':
            for s in people[p].get('children', dict()):
                for c in people[p]['children'][s]:
                    if not c:
                        continue
                    if c not in people:
                        people.update({c: {
                            'first': re.sub(f"([mf]-)?(\D+)({people[p].get('last', '').lower()})?\d*", '\g<2>',
                                            c).capitalize(),
                            'last': people[p].get('last', ''),
                            'gender': re.sub(f"([mf]?)-?(\w+)", '\g<1>', c)
                        }})
                        people[c]['shortname'] = joinName(people[c].get('first'), people[c].get('last'))
                    people[c]['father'] = p
                    people[c]['mother'] = s
        if people[p]['gender'] == 'F':
            for c in people[p].get('child', set()):
                people[c]['mother'] = p
    updateGenerationGroups(p0)


def runLinter(rawData: Dict) -> bool:
    ids: Set[str] = set()
    linted = True
    for person in rawData:
        if person['id'] in ids:
            warnings.warn(f"{person['id']} is a duplicate ID", Warning)
            linted = False
            continue
        ids.add(person['id'])
    return linted


def inFullTree(p: str, p0: str) -> bool:
    return p in getAncestors(p0) or p == p0


def getLineage(p: str, parent: str) -> str:
    if parent not in ['father', 'mother']:
        raise KeyError(f'{parent} is not a proper parent')
    p1 = getParent(p, parent)
    p2 = getParent(p, {'father': 'mother'}.get(parent, 'father'))
    if not p1 or p1 not in people:
        return ''
    parentLine = getLineage(p1, parent)
    line: str = f"\\namelink{{{p1}}}{{{people[p1]['first']}}}"
    line += ', ' if parentLine else ''
    line += parentLine
    return line


def getParent(p: str, parent: str) -> Union[str, None]:
    if parent not in ['father', 'mother']:
        raise KeyError(f'{parent} is not a proper parent')
    return people.get(p, {}).get(parent)


def printIndividualEntry(p: str, p0: str) -> str:
    # child_check(p)
    person = people[p]
    ancestor = getAncestorTag(person)
    name = getFullName(person)
    nameIndex = getNameIndex(person)
    title = getTitle(person)
    antonym = getAntonym(person)
    patriline = printLineage(p, 'father')
    birth = combineDatePlace(person, 'birth')
    death = combineDatePlace(person, 'death')
    vitals = combineVitals(birth, death)
    accolades = getAccolades(person)
    spouseDetails = generateSpouse(person, p0)
    history = person.get('history', None)
    childrenDetails = getChildrenDetails(person, p0)
    burialDetails = getBurialDetails(person)
    sources = getSources(person)

    return buildParagraphs(
        buildParagraph(
            buildSentence(
                f"\individual{ancestor}{{{p}}}{{{buildSentence(title, joinComma(name, antonym))}{nameIndex}}}",
                accolades,
                patriline,
                vitals
            ),
            *spouseDetails,
            history
        ),
        childrenDetails,
        burialDetails,
        sources
    )


def getSources(person: Dict) -> str:
    if not person.get('sources'):
        return ''
    allSources = person.get('sources')
    for s in person.get('spouse'):
        if not s:
            continue
        allSources |= people[s].get('sources', set())
    sources = [f"\\item{{{s}}}" for s in sorted(allSources)]
    return buildSentence('\\begin{source}', *sources, '\\end{source}')


def getBurialDetails(person: Dict) -> str:
    if person.get('buried'):
        return f"\\buried \\href{{{person.get('buriedlink') or ''}}}{{{person.get('buried')}}}"
    return ''


def getChildrenDetails(person: Dict, p0: str) -> str:
    childrens = []
    for s in person.get('children'):
        if not person.get('children')[s]:
            continue
        parentDetails = getParentDetails(person, s) + '\n'
        children = []
        for c in sorted(person.get('children')[s], key=lambda y: getVitalYear(y, 'birth') or 3000):
            if c not in people:
                warnings.warn(f"{c} doesn't have an entry", Warning)
                continue
            childDetail = getChildDetails(person, c, p0)
            children += [childDetail]
        if children:
            children.insert(0, parentDetails)
            childrens.append('\n'.join(children))
    return '\n'.join(childrens)


def getChildDetails(person: Dict, c: str, p0: str) -> str:
    mainLine = getMainLine(person, c)
    title = getTitle(people[c])
    antonym = getAntonym(people[c])
    birth = childBirth(c)
    marriage = childMarriage(c, mainLine, p0)
    return f"\\childlist{mainLine}{{{c if mainLine else ''}}}" \
           f"{{{buildSentence(title, joinComma(people[c]['shortname']), antonym)}}}" \
           f"{{{buildParagraph(birth, marriage)}}}"


def childMarriage(c: str, mainLine: str, p0: str) -> str:
    if people[c]['gender'] == 'M' or not mainLine:
        return ''
    spouses = people[c].get('spouse', '')
    if type(spouses) == str:
        spouses = [spouses]
    for cs in spouses:
        if not people.get(cs, dict()).get('generation'):
            continue
        return f"{getPronoun(people[c])} married {getShortNamelink(cs, p0)}"


def childBirth(c: str) -> str:
    if people[c].get('birthdate'):
        return f"born {people[c]['birthdate']}"
    return ''


def getMainLine(person: Dict, c: str) -> str:
    if c in person.get('child', set()):
        return '[+]'
    return ''


def getParentDetails(person: Dict, s: str) -> str:
    if not s:
        return f"{person['shortname']}\\children"
    if s not in people:
        return f"{person['shortname']} and {s}\\children"
    return f"{person['shortname']} and {people[s]['shortname']}\\children"


def generateSpouse(person: Dict, p0: str):
    spouse: list = person.get('spouse', [])
    if type(spouse) == str:
        spouse: list = [spouse]
    spouseDetail = []
    for s in sorted(spouse, key=lambda x: getMarriageYear(person, x)):
        if not s:
            continue
        spouseName = getSpouseName(s, p0)
        nSpouse = getSpouseNumber(s, spouse)
        marriage = combineMarriageDatePlace(person, s)
        birth = combineDatePlace(people[s], 'birth')
        death = combineDatePlace(people[s], 'death')
        vitals = combineVitals(birth, death, parents=getSpouseParents(s, p0))
        history = people[s].get('history')

        spouseDetail += [
            buildSentence(getPronoun(person), 'married', nSpouse, spouseName, marriage),
            buildSentence(getPronoun(people[s]) if vitals else None, vitals),
            history
        ]
    return spouseDetail


def getSpouseParents(s: str, p0: str) -> str:
    spouseFather = people[s].get('father')
    spouseMother = people[s].get('mother')
    spouseFatherName = ''
    spouseMotherName = ''
    if spouseFather in people:
        spouseFatherName = getShortNamelink(spouseFather, p0)
    if spouseMother in people:
        spouseMotherName = getShortNamelink(spouseMother, p0)
    return ' and '.join(filter(None, [spouseFatherName, spouseMotherName]))


def getShortNamelink(p: str, p0: str) -> str:
    if inFullTree(p, p0):
        return f"\\namelink{{{p}}}{{{people[p]['shortname']}}}"
    return f"{people[p]['shortname']}"


def getShortNamelinkBold(p: str, p0: str) -> str:
    if inFullTree(p, p0):
        return f"\\namelinkbold{{{p}}}{{{people[p]['shortname']}}}"
    return f"{people[p]['shortname']}"


def getSpouseName(s: str, p0: str) -> str:
    patriline = printLineage(s, 'father')
    shortName = people[s]['shortname']
    if inFullTree(s, p0):
        if not people[s].get('father') and not people[s].get('mother'):
            return f"\\textbf{{{shortName}}}{getNameIndex(people[s])}"
        return buildSentence(getShortNamelinkBold(s, p0), patriline, getNameIndex(people[s]))
    return f"\\textbf{{{shortName}}}"


def getSpouseNumber(s: str, spouse: iter) -> Union[int, str]:
    nSpouse = spouse.index(s) + 1
    if len(spouse) > 1:
        return f"({nSpouse})"
    return ''


def getPronoun(person: Dict) -> str:
    return {'M': 'He'}.get(person['gender'], 'She')


def buildParagraphs(*paragraphs: iter) -> str:
    return '\n\n'.join(filter(None, paragraphs))


def buildParagraph(*sentences: iter) -> str:
    return '. '.join(filter(None, sentences)) + '.'


def buildSentence(*phrases: iter) -> str:
    return ' '.join(filter(None, phrases))


def getAccolades(person: Dict) -> str:
    accolades = []
    for a in ['army', 'mason']:
        if person.get(a):
            accolades.append(a)
    return ''.join(f"\\{a}" for a in accolades)


def combineVitals(birth: str, death: str, parents: str = '') -> str:
    if birth:
        birth = f"was born {birth}"
    if parents:
        birth += f" to {parents}"
    if death:
        death = f"died {death}"
    vitals = '; '.join(filter(None, [birth, death]))
    return vitals


def combineDatePlace(person: Dict, vital: str) -> str:
    if vital not in ['birth', 'death', 'marriage']:
        raise KeyError(f'{vital} is not a vital statistic')
    vitalDateKey: str = f'{vital}date'
    vitalPlaceKey: str = f'{vital}place'
    vitalDate: str = person.get(vitalDateKey)
    vitalPlace: str = person.get(vitalPlaceKey)
    date: str = ''
    place: str = ''
    if vitalDate:
        full = isDateFull(vitalDate)
        date = f"{'on' if full else 'in'} {vitalDate}"
    if vitalPlace:
        place = f"in {vitalPlace}"
    return joinComma(date, place)


def combineMarriageDatePlace(person: Dict, s: str) -> str:
    vitalDateKey: str = 'marriagedate'
    vitalPlaceKey: str = f'marriageplace'
    vitalDate = person.get(vitalDateKey, {}).get(s)
    vitalPlace = person.get(vitalPlaceKey, {}).get(s)
    date: str = ''
    place: str = ''
    if vitalDate:
        full = isDateFull(vitalDate)
        date = f"{'on' if full else 'in'} {vitalDate}"
    if vitalPlace:
        place = f"in {vitalPlace}"
    return joinComma(date, place)


def joinComma(*phrases) -> str:
    return ', '.join(filter(None, phrases))


def printLineage(p, parent):
    patriline = getLineage(p, parent)
    if not patriline:
        return ''
    return f"({patriline})"


def getTitle(person: Dict) -> str:
    return person.get('title', '')


def getAntonym(person: Dict) -> str:
    return person.get('antonym', '')


def getNameIndex(person: Dict) -> str:
    firstName = person.get('first')
    middleName = person.get('middle')
    lastName = person.get('last')
    nameIndex = f"\index{{{lastName or ''}!{joinName(firstName, middleName)}}}"
    return nameIndex


def getFullName(person: Dict) -> str:
    firstName = person.get('first')
    middleName = person.get('middle')
    lastName = person.get('last')
    name = joinName(firstName, middleName, lastName)
    return name


def joinName(*name: iter) -> str:
    joinedName = ' '.join(filter(None, name))
    return joinedName


def getAncestorTag(person: Dict) -> str:
    if not person.get('father'):
        return '[p]'
    return ''


def descent(p: str, p0: str) -> list:
    d = [[p]]
    while d[0][-1] != p0:
        children = list(people[d[0][-1]]['child'])
        if len(children) > 1:
            d.append(d[-1] + descent(children[1], p0)[0])
        d[0].append(children[0])

    return sorted(d, key=lambda m: lenSpearLine(m[::-1]), reverse=True)


def lenSpearLine(m: list) -> int:
    i = int(people[m[0]]['gender'] == 'M')
    c = 0
    for j in m[i:]:
        c += int(people[j]['gender'] == 'M')
    return c


def updateGenerationGroups(p0):
    generations.clear()
    setGenerations(p0)
    maxG = max(people[p].get('generation', 0) for p in people)
    for g in range(maxG + 1):
        generations.insert(0, set(filter(lambda p: isInGeneration(g, p), people)))


def setGenerations(p0):
    for p in filter(lambda p: inFullTree(p, p0), people):
        people[p]['generation'] = max(len(d) - 1 for d in descent(p, p0))


def isInGeneration(g: int, p: str) -> bool:
    return 'generation' in people[p] and people[p]['generation'] == g


def getAncestors(p):
    ancestors = {p}
    i = 0
    while i != len(ancestors):
        i = len(ancestors)
        for q in list(ancestors):
            if people[q].get('father') in people:
                ancestors.add(people[q].get('father'))
            if people[q].get('mother') in people:
                ancestors.add(people[q].get('mother'))
    ancestors.discard(p)
    return ancestors


def child_check(p):
    for s in people[p]['children']:
        for c in people[p]['children'][s]:
            if people.get(c, dict()).get('generation', 100) >= people[p].get('generation', 0):
                print(f'generation issue: {p} and {c}')


def getVitalYear(p: str, vital: str) -> Optional[int]:
    if vital not in ['birth', 'death']:
        raise KeyError(f"{vital} is not a vital statistic")
    date: Union[int, str] = people.get(p, {}).get(f'{vital}date', 0)
    if not date:
        return None
    if type(date) == int:
        return date
    date = people[p][f'{vital}date'].split(' ')
    return int(date[-1])


def getMarriageYear(person: Dict, s: str) -> int:
    if not s:
        return 0
    date: Union[int, str] = person.get('marriagedate', {}).get(s, 0)
    if not date:
        return 0
    if type(date) == int:
        return date
    date: list = person.get('marriagedate', {}).get(s, '').split(' ')
    return int(date[-1])


def generateTex(familyName: str, p0: str):
    importFamily(familyName, p0)
    with open(fr"{os.getcwd()}\{familyName}_generated.tex", 'w') as f:
        writeTitle(f, familyName)
        writeGenerations(f, p0)
    # h_tree(file, p0)
    # line_chart(file, p0)
    # follow(lost=True)
    # unsourced()
    print(f"{len(set.union(*generations))} total ancestors")


def writeGenerations(f, p0: str):
    for g in generations:
        writeGeneration(f, p0, g)


def writeGeneration(f, p0: str, g: Set):
    if p0 in g and people[p0]['gender'] == 'F':
        return
    f.write(f"\\generationgroup\n\n")
    for p in sorted(list(g), key=lambda y: getVitalYear(y, 'birth') or 3000):
        if people[p]['gender'] == 'F' and people[p].get('spouse', []):
            continue
        f.write(f"{printIndividualEntry(p, p0)}\n\n")


def writeTitle(f, familyName: str):
    f.write(f"\\chapter*{{{familyName}}}\n\n")


def follow(p0: str, g=None, lost=False):
    i = 0
    endOfLine = dict()
    for p in sorted(people, key=lambda p: getVitalYear(p, 'birth') or 3000):
        if lost and people[p].get('lost'):
            continue
        if g and people[p].get('generation') != g:
            continue
        if people[p]['gender'] == 'F' and people[p].get('spouse') and people[p].get('generation'):
            if type(people[p].get('spouse')) == list:
                spouseNote = ', '.join([people[s].get('note') for s in people[p]['spouse'] if s])
            else:
                spouseNote = people[people[p]['spouse']].get('note')
            if not people[p].get('father'):
                endOfLine.update({
                    p: {'spouse': people[p]['spouse'], 'birthyear': getVitalYear(p, 'birth'),
                        "note": people[p].get('note'), 'spouseNote': spouseNote}
                })
                i += 1
            elif not inFullTree(people[p].get('father'), p0):
                endOfLine.update({
                    p: {'spouse': people[p]['spouse'], 'birthyear': getVitalYear(p, 'birth'),
                        "note": people[p].get('note'), 'spouseNote': spouseNote}
                })
                i += 1
        if people[p]['gender'] == 'M' and people[p].get('generation') and not inFullTree(people[p].get('father'), p0):
            endOfLine.update({
                p: {'birthyear': getVitalYear(p, 'birth'), "note": people[p].get('note')}
            })
        i += 1
    for line in sorted(endOfLine, key=lambda p: endOfLine[p].get('birthyear') or 0):
        print(line, endOfLine[line])
    print(f"{i} threads to pull")


def unsourced(g=None):
    i = 0
    for g in generations[::-1][2:]:
        for p in g:
            if people[p]['gender'] == 'M' and not people[p]['sources']:
                print(p, people[p]['shortname'], getVitalYear(p, 'birth'), people[p]['note'])
                i += 1
    print(f"{i} sources to get")
    # return


def generation_count(children=False):
    if children:
        for g in generations:
            N = len(set.union(*(
                [set.union(
                    *[people[p]['children'][s] for s in people[p]['children']] if people[p]['children'] else [set()])
                    for p in g])))
            print(f"{generations.index(g) + 1}\t{N}\t{len(g)}")
    else:
        for g in generations:
            print(f"{generations.index(g) + 1}\t{len(g)}")


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
    tree = {root: "parent{g{" + f"{people[root]['first']} {people[root]['last'] or '---'}" + "}}"}
    main_people = set()
    for g in generations:
        for p in g:
            tree.update({p: "parent{g{" + f"{people[p]['first']} {people[p]['last'] or '---'}" + "}}"})
            main_people.add(p)
    done = set()
    collapsed_tree = tree.copy()
    for g in generations:
        for c in sorted(set.union(*[people[p]['child'] for p in g]), key=lambda i: people[i]['gender'], reverse=True):
            if main and c not in main_people:
                continue
            if c in done:
                continue
            father, mother, siblings = '', '', ''
            if people[c]['father'] in g:
                if people[c]['father'] in done:
                    father = tree[people[c]['father']]
                else:
                    father = collapsed_tree[people[c]['father']]
                done.add(people[c]['father'])
            if people[c]['mother'] in g:
                if people[c]['mother'] in done:
                    mother = tree[people[c]['mother']]
                else:
                    mother = collapsed_tree[people[c]['mother']]
                done.add(people[c]['mother'])
                try:
                    for s in sorted(people[people[c]['father']]['children'][people[c]['mother']],
                                    key=lambda p: people[p]['gender'], reverse=True):
                        # for s in sorted(people[people[c]['father']]['children'][people[c]['mother']], key=lambda p: get_birth_year(p), reverse=True):
                        if s == c:
                            continue
                        if main and s not in main_people:
                            continue
                        siblings += "c{" + people[s]['shortname'] + "}"
                        done.add(s)
                except:
                    pass
            collapsed_tree[c] = collapsed_tree[c][:-1] + f"{siblings}{father}{mother}" + collapsed_tree[c][-1:]
    print(collapsed_tree[root])


def generateHTree(file: str, p: str, size: int = 90):
    N = len(generations) // 2
    spacing: int = size + int(size / 6)

    ancestors, descendants, initialPositions = getInitialPositionsH(size, N, p)
    compaction = [horizontalCompactionLTR, horizontalCompactionRTL, verticalCompactionTTB, verticalCompactionBTT]
    while True:
        oldPositions = copy.deepcopy(initialPositions)
        compaction = compaction[::-1]
        for outerEdge in [True, False]:
            for f in compaction:
                initialPositions = f(initialPositions, descendants, ancestors, outerEdge)
                initialPositions = compactTwigsLeaves(ancestors, descendants, initialPositions, spacing)
        if getHArea(oldPositions, descendants)['area'] <= getHArea(initialPositions, descendants)['area']:
            initialPositions = copy.deepcopy(oldPositions)
            print(getHArea(initialPositions, descendants))
            break

    area = getHArea(initialPositions, descendants)
    print(area)
    drawHTree(area, descendants, file, initialPositions, size, p)


def compactTwigsLeaves(ancestors, descendants, initialPositions, spacing):
    initialPositions = verticalCompactionLeaves(initialPositions, descendants, ancestors, spacing)
    initialPositions = horizontalCompactionLeaves(initialPositions, descendants, ancestors, spacing)
    initialPositions = horizontalCompactionTwig(initialPositions, descendants, ancestors, spacing)
    initialPositions = verticalCompactionTwig(initialPositions, descendants, ancestors, spacing)
    return initialPositions


def verticalCompactionLeaves(initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int) -> Dict:
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


def horizontalCompactionLeaves(initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int) -> Dict:
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


def verticalCompactionTwig(initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int) -> Dict:
    bars = getYBars(ancestors, descendants, initialPositions)
    visibilityTTB = getVisibilityTTB(bars, initialPositions)
    visibilityBTT = getVisibilityBTT(bars, initialPositions)
    for b in bars:
        if len(bars[b]) == 3 and bars[b][0] not in ancestors and bars[b][-1] not in ancestors:
            p = bars[b][1]
        elif len(bars[b]) == 2 and any(i not in ancestors for i in bars[b]):
            p = bars[b][0] if bars[b][1] not in ancestors else bars[b][1]
        else:
            continue
        childDist = getChildDist(descendants, initialPositions, p)
        if childDist == spacing:
            continue
        c = descendants[p]
        if c in bars.get(visibilityTTB.get(b, ()), []) + bars.get(visibilityBTT.get(b, ()), []):
            yc = initialPositions[c][1]
            for node in bars[b]:
                xa, ya = initialPositions[node]
                initialPositions[node] = (xa, yc + spacing * {0: -1, 1: 1}[yc < ya])
    return initialPositions


def horizontalCompactionTwig(initialPositions: Dict, descendants: Dict, ancestors: Dict, spacing: int) -> Dict:
    bars = getXBars(ancestors, descendants, initialPositions)
    visibilityLTR = getVisibilityLTR(bars, initialPositions)
    visibilityRTL = getVisibilityRTL(bars, initialPositions)
    for b in bars:
        if len(bars[b]) == 3 and bars[b][0] not in ancestors and bars[b][-1] not in ancestors:
            p = bars[b][1]
        elif len(bars[b]) == 2 and any(i not in ancestors for i in bars[b]):
            p = bars[b][0] if bars[b][1] not in ancestors else bars[b][1]
        else:
            continue
        childDist = getChildDist(descendants, initialPositions, p)
        if childDist == spacing:
            continue
        c = descendants[p]
        if c in bars.get(visibilityLTR.get(b, ()), []) + bars.get(visibilityRTL.get(b, ()), []):
            xc = initialPositions[c][0]
            for node in bars[b]:
                xa, ya = initialPositions[node]
                initialPositions[node] = (xc + spacing * {0: -1, 1: 1}[xc < xa], ya)
    return initialPositions


def drawHTree(area, descendants, file, initialPositions, size, p0):
    with open(f"{file}-H.svg", 'r') as f:
        svg = bs4.BeautifulSoup(f, 'xml')
    treeStyle = f"fill:burlywood;"
    pStyle = f"stroke:black;stroke-width:2px;"
    gen = lambda g: f"fill:white;opacity:{g / len(generations):0.2f};"
    nameStyle = 'fill:black;text-anchor:middle;alignment-baseline:middle;font-size:9pt;font-family:Chomsky;'
    duplicate = 'opacity:0.5;'
    linesStyle = 'stroke:black;stroke-width:5px;'
    styles = f"<style type='text/css'>" \
             f".tree {{{treeStyle}}} " \
             f".name {{{nameStyle}}} " \
             f".duplicate {{{duplicate}}} " \
             f".line {{{linesStyle}}} " \
             f".tree {{{treeStyle}}}" \
             f".primary {{{pStyle}}} " \
             f"</style>"
    tree = ''
    names = ''
    lines = ''
    minx = area['minx']
    maxx = area['maxx']
    miny = area['miny']
    maxy = area['maxy']
    for p in initialPositions:
        x, y = initialPositions[p]
        xc, yc = initialPositions[descendants[p]]
        key = p[:-2]
        rx = {'M': int(size / 6), 'F': int(size / 2)}[people[key]['gender']]
        tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' rx='{rx}' width='{size}' height='{size}' class='tree{' primary' if p0 == p[:-2] else ''}' id='{p}' />"
        tree += f"<rect x='{x - size / 2}' y='{y - size / 2}' width='{size}' height='{size}' style='{gen(people[key]['generation'])}'/>"
        if people[key].get('last'):
            names += f"<text class='name{' duplicate' if int(p[-1]) != 0 else ''}'><tspan x='{x}' y='{y}' dy='-2.5pt'>{people[key]['first']}</tspan><tspan x='{x}' y='{y}' dy='7.5pt'>{people[key].get('last', '')}</tspan></text>"
        else:
            names += f"<text class='name{' duplicate' if int(p[-1]) != 0 else ''}'><tspan x='{x}' y='{y}'>{people[key]['first']}</tspan></text>"
        lines += f"<path d='M {x},{y} L {xc},{yc}' class='line'/>"
    svg.find('svg').clear()
    svg.find('svg').contents = bs4.BeautifulSoup(styles, 'html.parser').contents
    svg.find('svg').contents += bs4.BeautifulSoup(lines, 'html.parser').contents
    svg.find('svg').contents += bs4.BeautifulSoup(tree, 'html.parser').contents
    svg.find('svg').contents += bs4.BeautifulSoup(names, 'html.parser').contents
    svg.find('svg').attrs.update(
        {'viewBox': f"{minx - size / 2} {miny - size / 2} {maxx - minx + size} {maxy - miny + size}"})
    with open(f"{file}-H.svg", 'w') as f:
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
                key = '-'.join([c, str(int(c in done))])
                initialPositions.update({key: (x, y)})
                xc, yc = positionH(d[1:], N=N)
                done.add(c)
                keyc = [i for i in initialPositions if initialPositions[i] == (xc, yc)]
                if not keyc:
                    continue
                keyc = keyc[0]
                descendants.update({key: keyc})
                ancestors.setdefault(keyc, set())
                ancestors[keyc].add(key)
            if people[c].get('father') in people:
                current.add(people[c]['father'])
            if people[c].get('mother') in people:
                current.add(people[c]['mother'])
            current.discard(c)
    return ancestors, descendants, initialPositions


def positionH(d, N, s=90 + 15) -> Tuple[int, int]:
    x, y = 0, 0
    if len(d) == 1:
        return x, y
    for i, j in enumerate(d[::-1][1:]):
        if not (i + 1) % 2:
            y += (-1 if people[j]['gender'] == 'M' else 1) * s * 2 ** (N - people[j]['generation'] // 2)
        else:
            x += (-1 if people[j]['gender'] == 'M' else 1) * s * 2 ** (N - people[j]['generation'] // 2 - 1)
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
    return {'maxx': maxx, 'maxy': maxy, 'minx': minx, 'miny': miny, 'area': area}


def getChildDist(descendants: Dict, initialPositions: Dict, p: str) -> int:
    c = descendants[p]
    xp, yp = initialPositions[p]
    xc, yc = initialPositions[c]
    edge = abs((xc - xp) + (yc - yp))
    return edge


def horizontalCompactionLTR(initialPositions: Dict, descendants: Dict, ancestors: Dict,
                            outerEdge: bool = False) -> Dict:
    bars = getXBars(ancestors, descendants, initialPositions)
    # Create visibility graph
    # Update bar location
    spacing = 90 + 15
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True):
        visibility = getVisibilityLTR(bars, initialPositions)
        if b not in visibility:
            c = sorted(bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True)[0]
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


def verticalCompactionBTT(initialPositions: Dict, descendants: Dict, ancestors: Dict, outerEdge: bool = False) -> Dict:
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
        for c in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True):
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


def horizontalCompactionRTL(initialPositions: Dict, descendants: Dict, ancestors: Dict,
                            outerEdge: bool = False) -> Dict:
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
        for c in sorted(bars, key=lambda b: initialPositions[bars[b][0]][0], reverse=True):
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


def verticalCompactionTTB(initialPositions: Dict, descendants: Dict, ancestors: Dict, outerEdge: bool = False) -> Dict:
    bars = getYBars(ancestors, descendants, initialPositions)
    spacing = 90 + 15
    for b in sorted(bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True):
        visibility = getVisibilityTTB(bars, initialPositions)
        if b not in visibility:
            c = sorted(bars, key=lambda b: initialPositions[bars[b][0]][1], reverse=True)[0]
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


def getXBars(ancestors: Dict, descendants: Dict, initialPositions: Dict) -> Dict[Tuple[int, int], List[str]]:
    xPositions: Set[int] = {initialPositions[i][0] for i in initialPositions}
    # Create axis dictionary
    axis: Dict = {}
    for x in sorted(xPositions):
        nodes: Iterable = sorted({i for i in initialPositions if initialPositions[i][0] == x},
                                 key=lambda n: initialPositions[n][1])
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


def getYBars(ancestors: Dict, descendants: Dict, initialPositions: Dict) -> Dict[Tuple[int, int], List[str]]:
    yPositions: Set[int] = {initialPositions[i][1] for i in initialPositions}
    # Create axis dictionary
    axis: Dict = {}
    for y in sorted(yPositions):
        nodes: Iterable = sorted({i for i in initialPositions if initialPositions[i][1] == y},
                                 key=lambda n: initialPositions[n][0])
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
    current: Set[str] = {root}
    done: Dict[str, Dict[str, int]] = dict()
    while current:
        for p in list(current):
            b = getVitalYear(p, 'birth')
            d = getVitalYear(p, 'death')
            y = 0
            done.update({p: {'b': b, 'd': d, 'y': y}})
            if people[p].get('father') in people:
                current.add(people[p]['father'])
            if people[p].get('mother') in people:
                current.add(people[p]['mother'])
            current.discard(p)

    unknownb: Set[str] = set()
    unknownd: Set[str] = set()
    for p in done:
        if not getVitalYear(p, 'death'):
            if getVitalYear(p, 'birth') and currentYear - getVitalYear(p, 'birth') < 100:
                done[p]['d'] = currentYear
                continue
            unknownd.add(p)
            done[p]['d'] = getApproxDeath(done, p)
        if not getVitalYear(p, 'birth'):
            unknownb.add(p)
            done[p]['b'] = getApproxBirth(done, p)

    numGenerations = len(generations)
    ancestors, descendants, initialPositions = getInitialPositionsLine(pt2px(10), numGenerations, root)
    positions = verticalCompactionLineTTB(done, initialPositions, descendants, ancestors)
    positions = verticalCompactionLineBTT(done, initialPositions, descendants, ancestors)
    positions = verticalCompactionLineTTB(done, initialPositions, descendants, ancestors)
    for p in done:
        done[p]['y'] = positions[p]
    minY = min(done[p]['y'] for p in done)
    for p in done:
        done[p]['y'] = done[p]['y'] - minY

    drawLineChart(currentYear, done, file, unknownb, unknownd)


def drawLineChart(currentYear: int, done: dict, file: str, unknownb: set, unknownd: set):
    def addPt(i: str, event: str, pt: int = 0) -> float:
        return done[i][event] + pt / 0.75

    minx = min(done[p]['b'] for p in done) - 10
    maxx = currentYear
    widx = maxx - minx
    w = 18 * 96
    miny = min(done[p]['y'] for p in done) - 10
    maxy = max(done[p]['y'] for p in done) + 10
    with open(fr"{os.getcwd()}\{file}-lines.svg", 'r') as f:
        svg = bs4.BeautifulSoup(f, 'xml')
    lines = ''
    begats = ''
    names = ''
    years = ''
    flags = ''
    for p in done:
        if people[p].get('mother') in done:
            m = people[p]['mother']
            begats += f"<path d='M {done[p]['b'] * w / widx:.3f},{addPt(p, 'y', -5):.3f} " \
                      f"V {addPt(m, 'y', 5):.3f}' " \
                      f"class='{'unbegat' if p in unknownb | unknownd else 'begat'}' id='{m}-{p}'/>"
            begats += f"<circle cx='{done[p]['b'] * w / widx:.3f}' cy='{addPt(m, 'y', -5):.3f}' r='3' />"
        if people[p].get('father') in done:
            f = people[p]['father']
            begats += f"<path d='M {done[p]['b'] * w / widx:.3f},{addPt(p, 'y', 5):.3f} " \
                      f"V {addPt(f, 'y', -5):.3f}' " \
                      f"class='{'unbegat' if p in unknownb | unknownd else 'begat'}' id='{f}-{p}'/>"
            begats += f"<circle cx='{done[p]['b'] * w / widx:.3f}' cy='{addPt(f, 'y'):.3f}' r='3' />"
        if people[p].get('mother') in done or people[p].get('father') in done:
            begats += f"<circle cx='{done[p]['b'] * w / widx:.3f}' cy='{addPt(p, 'y'):.3f}' r='3' />"
    for p in done:
        if p in unknownb and p not in unknownd:
            p_class = 'unknownb'
            if getState(p, 'death'):
                flags += f"<image x='{(done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{getState(p, 'death').lower()}.png'/>"
        elif p not in unknownb and p in unknownd:
            p_class = 'unknownd'
            if getState(p, 'birth'):
                flags += f"<image x='{(done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{getState(p, 'birth').lower()}.png' style='transform: translateX(-15.72pt)'/>"
        elif p in unknownb and p in unknownd:
            p_class = 'unknownbd'
        else:
            p_class = 'known'
            if getState(p, 'death'):
                flags += f"<image x='{(done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{getState(p, 'death').lower()}.png'/>"
            if getState(p, 'birth'):
                flags += f"<image x='{(done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{getState(p, 'birth').lower()}.png' style='transform: translateX(-15.72pt)'/>"
        lines += f"<rect x='{done[p]['b'] * w / widx:.3f}' y='{addPt(p, 'y', -5):.3f}' " \
                 f"width='{(done[p]['d'] - done[p]['b']) * w / widx:.3f}' height='{pt2px(10):.3f}' id='{p}'" \
                 f"class='{p_class}' />"
        if p in unknownb:
            lines += f"<rect x='{(done[p]['b'] - 5) * w / widx:.3f}' y='{addPt(p, 'y', -5):.3f}' " \
                     f"width='{6 * w / widx:.3f}' height='{pt2px(10):.3f}' id='{p}-b'" \
                     f"class='unknownbb' />"
        if p in unknownd:
            lines += f"<rect x='{(done[p]['d'] - 1) * w / widx:.3f}' y='{addPt(p, 'y', -5):.3f}' " \
                     f"width='{6 * w / widx:.3f}' height='{pt2px(10):.3f}' id='{p}-d'" \
                     f"class='unknownda' />"
        for s in people[p].get('marriageplace', []):
            if not getMarriageYear(people[p], s):
                continue
            flags += f"<image x='{getMarriageYear(people[p], s) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                     f"height='9pt' href='flags/{getState(p, 'marriage')[s].lower()}.png' style='transform: translateX(-7.36pt)'/>"
    for p in done:
        names += f"<text class='name {'nameunknown' if p in unknownb | unknownd else ''}'>" \
                 f"<tspan dx='{pt2px(2):.3f}' dy='{pt2px(1):.3f}' x={done[p]['b'] * w / widx:.3f} y={done[p]['y']:.3f}>" \
                 f"{people[p]['first']} {people[p].get('last', '')}</tspan></text>"
    for y in range(math.floor(minx), math.ceil(maxx) + 1):
        if not y % 100:
            years += f"<path d='M {y * w / widx:.0f},{maxy} V {miny}' class='year' id='{y}'/>"
    minx = min((done[p]['b'] - 10) * w / widx for p in done)
    maxx = currentYear * w / widx
    svg.find('svg').find('g').clear()
    svg.find('svg').find('g').contents = bs4.BeautifulSoup(
        '<g id="yearLines"/><g id="begats"/><g id="lines"/><g id="flags" style="display:none"/><g id="names"/>',
        'html.parser'
    ).contents
    svg.find('g').find('g', id="yearLines").contents = bs4.BeautifulSoup(years, 'html.parser').contents
    svg.find('g').find('g', id="begats").contents = bs4.BeautifulSoup(begats, 'html.parser').contents
    svg.find('g').find('g', id="lines").contents = bs4.BeautifulSoup(lines, 'html.parser').contents
    svg.find('g').find('g', id="flags").contents = bs4.BeautifulSoup(flags, 'html.parser').contents
    svg.find('g').find('g', id="names").contents = bs4.BeautifulSoup(names, 'html.parser').contents
    svg.find('svg').attrs.update({'viewBox': f"{minx - 10} {miny - 10} {maxx - minx + 20} {maxy - miny + 20}"})
    svg.find('svg').attrs.update({'height': f"{maxy - miny}", 'width': f"{maxx - minx}"})
    svg.find('style').string = ".known {fill:#888;} " \
                               ".unknownb {fill:url(#ub);} " \
                               ".unknownd {fill:url(#ud);} " \
                               ".unknownbd {fill:url(#ubd);} " \
                               ".unknownda {fill:url(#uda);} " \
                               ".unknownbb {fill:url(#ubb);} " \
                               ".begat {stroke-width:1pt; stroke:#888} " \
                               "#begats circle {fill:#888} " \
                               ".unbegat {stroke-width:1pt; stroke:#aaa} " \
                               ".name {dominant-baseline:middle; paint-order:stroke fill; font: bold 9pt Carlito; fill:black; stroke: white;stroke-width:2px} " \
                               ".nameunknown {fill:#333}" \
                               ".year {stroke-dasharray:5pt; stroke-width:2pt; stroke:#ccc}"
    with open(fr"{os.getcwd()}\{file}-lines.svg", 'w') as f:
        f.write(svg.prettify())


def getInitialPositionsLine(size: float, numGenerations: int, p0: str) -> Tuple[Dict, Dict, Dict]:
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
            if people[c].get('father') in people:
                current.add(people[c]['father'])
            if people[c].get('mother') in people:
                current.add(people[c]['mother'])
            current.discard(c)
    return ancestors, descendants, initialPositions


def getVisibilityLineBTT(done: dict, bars: dict, positions: dict) -> dict:
    visibility: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in sorted(bars, key=lambda b: positions[bars[b][0]], reverse=True):
        y = b[0]
        x1 = min([done[i]['b'] for i in bars[b]]) - 10
        x2 = max([done[i]['d'] for i in bars[b]]) + 10
        for c in sorted(bars, key=lambda b: positions[bars[b][0]], reverse=True):
            if c[0] >= y:
                continue
            x3 = min([done[i]['b'] for i in bars[c]]) - 10
            x4 = max([done[i]['d'] for i in bars[c]]) + 10
            if x3 <= x1 <= x4 or x3 <= x2 <= x4:
                visibility[b] = c
                break
            if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                visibility[b] = c
                break
    return visibility


def verticalCompactionLineBTT(done: dict, positions: Dict, descendants: Dict, ancestors: Dict,
                              outerEdge: bool = False) -> Dict:
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


def verticalCompactionLineTTB(done: dict, positions: Dict, descendants: Dict, ancestors: Dict,
                              outerEdge: bool = False) -> Dict:
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
        x1 = min([done[i]['b'] for i in bars[b]]) - 10
        x2 = max([done[i]['d'] for i in bars[b]]) + 10
        for c in sorted(bars, key=lambda b: positions[bars[b][0]]):
            if c[0] <= y:
                continue
            x3 = min([done[i]['b'] for i in bars[c]]) - 10
            x4 = max([done[i]['d'] for i in bars[c]]) + 10
            if x3 <= x1 <= x4 or x3 <= x2 <= x4:
                visibility[b] = c
                break
            if x1 <= x3 <= x2 or x1 <= x4 <= x2:
                visibility[b] = c
                break
    return visibility


def getYBarsLine(done: Dict, ancestors: Dict, descendants: Dict, initialPositions: Dict) -> Dict[
    Tuple[int, int], List[str]]:
    yPositions: Set[int] = {initialPositions[i] for i in initialPositions}
    # Create axis dictionary
    axis: Dict = {}
    for y in sorted(yPositions):
        nodes: Iterable = sorted({i for i in initialPositions if initialPositions[i] == y}, key=lambda n: done[n]['b'])
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
        yy = round((1 if people[j]['gender'] == 'F' else -1) * size * 2 ** (numGenerations - people[j]['generation']))
        y += yy
    return y


def getApproxDeath(done: dict, p: str) -> int:
    if people[p].get('deathdate'):
        return getVitalYear(p, 'death')
    allChildren = getAllChildren(p)
    lastChildBirth: int = max(list(filter(None, [getApproxBirth(done, c) for c in allChildren])) or [None])
    if people[p]['gender'] == 'M':
        spouses = people[p].get('spouse', [])
        marriageDates: List[Optional[int]] = [getMarriageYear(people[p], s) for s in spouses]
    else:
        spouse = people[p].get('spouse')[0] if type(people[p].get('spouse')) == list else people[p].get('spouse')
        marriageDates: List[Optional[int]] = [getMarriageYear(people[p], spouse)]
    marriageDates: Optional[int] = max(list(filter(lambda x: x != 0, marriageDates)) or [None])
    return max(list(filter(None, [lastChildBirth, marriageDates])) or [None])


def getApproxBirth(done: dict, p: str) -> int:
    if people[p].get('birthdate'):
        return getVitalYear(p, 'birth')
    allChildren = getAllChildren(p)
    firstChildBirth: Optional[int] = min(list(filter(None, [getApproxBirth(done, c) for c in allChildren])) or [None])
    if firstChildBirth:
        firstChildBirth -= 20
    fatherDeath: Optional[int] = done.get(people[p].get('mother'), dict()).get('d')
    motherDeath: Optional[int] = done.get(people[p].get('father'), dict()).get('d')
    if people[p]['gender'] == 'M':
        marriageDates: List[Optional[int]] = [getMarriageYear(people[p], s) for s in people[p].get('spouse', [])]
    else:
        marriageDates: List[Optional[int]] = [getMarriageYear(people[p], people[p].get('spouse'))]
    marriageDates: Optional[int] = min(list(filter(lambda x: x != 0, marriageDates)) or [None])
    if marriageDates:
        marriageDates -= 20
    return min(list(filter(None, [fatherDeath, motherDeath, firstChildBirth, marriageDates])) or [None])


def getAllChildren(p: str) -> Set[str]:
    allChildren = {i for i in people if p in {people[i].get('mother'), people[i].get('father')}}
    return allChildren


def birthdays(month, p0):
    for p in people:
        if inFullTree(p, p0) and month in str(people[p]['birthdate']):
            announce(p, p0)
            print(' ')


def announce(p, p0):
    for d in descent(p, p0):
        a = f"On {people[p]['birthdate']}, {people[p]['shortname']} was born"
        if people[p]['birthplace']:
            a += f" in {people[p]['birthplace']}.{' ' + people[p]['history'] if people[p]['history'] else ''}\n"
        for n, i in enumerate(d[:-1]):
            if people[i]['title']:
                a += f"{people[i]['title']} "
            if people[i]['spouse']:
                if people[i]['gender'] == 'M':
                    for s in people[i]['spouse']:
                        if d[n + 1] in people[i]['children'][s]:
                            break
                else:
                    s = people[i]['spouse'][0] if type(people[i]['spouse']) == list else people[i]['spouse']
                if s:
                    a += f"{people[i]['shortname']} married {people[s]['shortname']} and begat {people[d[n + 1]]['shortname']}.\n"
                else:
                    a += f"{people[i]['shortname']} begat {people[d[n + 1]]['shortname']}.\n"
            else:
                a += f"{people[i]['shortname']} begat {people[d[n + 1]]['shortname']}.\n"
        a += f"{people[p]['first']} was my {'great-' * (len(d) - 2)}grand{'father' if people[p]['gender'] == 'M' else 'mother'}."
        if people[p]['buried']:
            a += f"\n{'He' if people[p]['gender'] == 'M' else 'She'} is buried in {people[p]['buried']}."
        print(a)


def getLivingAncestors(p: str) -> Set[str]:
    b0 = getVitalYear(p, 'birth')
    d0 = getVitalYear(p, 'death')
    alive = set()
    if not b0 and not d0:
        raise ValueError(f'{p} doesnt have vital statistics')
    year = datetime.date.today().year
    for a in getAncestors(p):
        b = getVitalYear(a, 'birth')
        d = getVitalYear(a, 'death')
        if not d and year - b < 100:
            d = year
        if b0 and d and d > b0:
            alive.add(a)
    return alive


def getState(p: dict, time: str) -> Union[None, dict, str]:
    if not people[p].get(f'{time}place'):
        return None
    if time == 'marriage':
        state = {m: people[p].get(f'{time}place', {m: ','})[m].split(',')[-1].strip() for m in
                 people[p].get(f'{time}place', {'': ','})}
        return state
    state = people[p].get(f'{time}place', ',').split(',')
    return state[-1].strip()


def ancestors_by_age():
    byAge = [p for p in set.union(*generations) if people[p]['birthdate'] and people[p]['deathdate']]
    for p in sorted(byAge, key=lambda p: getVitalYear(p, 'death') - getVitalYear(p, 'birth')):
        print(people[p]['shortname'], getVitalYear(p, 'death') - getVitalYear(p, 'birth'))


def pt2px(pt: Union[float, int]) -> float:
    return pt / 0.75


def generateID(name: dict) -> dict:
    nameID = (name.get('first', '') + name.get('last', '')).replace(' ', '').lower()
    nameID = nameID.replace('.', '').replace("'", '')
    name.update({"id": generateIDn(nameID)})
    return name


def generateIDn(nameID: str) -> str:
    i = 1
    while True:
        if (nameIDi := nameID + str(i) if i > 1 else nameID) not in people:
            return nameIDi
        i += 1


def getSpouse(p: str) -> List[str]:
    if len(people[p]['spouse']) == 0:
        return ''
    if type(people[p]['spouse']) == str:
        return [people[p]['spouse']]
    return people[p]['spouse']


def getChildren(p: str, spouse: str):
    children = list(people[p]['children'].get(spouse, set()) |
                    people.get(spouse, dict()).get('children', dict()).get(p, set()))
    children.sort(key=lambda c: (getVitalYear(c, 'birth') is None, getVitalYear(c, 'birth')))
    return children


def drawZegelchart(p: str):
    currentYear: int = datetime.datetime.now().year
    descendants = generateZegelchart(p)
    toPt = lambda pt: pt / 0.75
    width = lambda p: (d['d'] or 2023) - d['b']
    y = lambda i: (2 * i) * toPt(10)
    xMin = min(d['b'] for d in descendants if d['b']) - 10
    xMax = currentYear + 10
    xWidth = xMax - xMin
    yMin = 0
    yMax = len(descendants) * 2
    with open(fr"{os.getcwd()}\{p}-zegelchart.svg", 'r') as f:
        svg = bs4.BeautifulSoup(f, 'xml')
    lives = ''
    lines = ''
    names = ''
    done = set()
    for i, d in enumerate(descendants):
        if not d['b']: continue
        lives += f"<rect id='{d['id']}' height='10pt' width='{width(d)}' x='{d['b']:.3f}' y='{y(i):.3f}' />"
        for s in getSpouse(d['id']):
            if s not in done:
                sidx = next((j for j, e in enumerate(descendants) if e['id'] == s), None)
                lines += f"<path id='{d['id']}-{s}' d='M {getMarriageYear(people[d['id']], s):.3f} {y(i) + toPt(5):.3f} V {y(sidx) + toPt(5):.3f}' />"
        if next((j for j, e in enumerate(descendants) if e['id'] == getParent(d['id'], 'father')), None) is not None:
            if marriageYear := getMarriageYear(people[getParent(d['id'], 'father')], getParent(d['id'], 'mother')):
                lines += f"<path id='{d['id']}-parent' d='M {d['b']:.3f} {y(i) + toPt(5):.3f} H {marriageYear:.3f}' />"
        names += f"<text x='{currentYear + toPt(2):.3f}' y='{y(i)+toPt(5):.3f}'>{people[d['id']]['shortname']}</text>"
        done.add(d['id'])
    svg.find('g', id="lives").contents = bs4.BeautifulSoup(lives, 'html.parser').contents
    svg.find('g', id="lines").contents = bs4.BeautifulSoup(lines, 'html.parser').contents
    svg.find('g', id="names").contents = bs4.BeautifulSoup(names, 'html.parser').contents
    svg.find('svg').attrs.update(
        {'viewBox': f"{xMin - 10:.3f} {yMin - 10:.3f} {xWidth + 20:.3f} {(yMax - yMin) * toPt(10) + 10:.3f}"})
    with open(fr"{os.getcwd()}\{p}-zegelchart.svg", 'w') as f:
        f.write(svg.prettify())


def generateZegelchart(p: str) -> List[Dict[str, Union[int, str, None]]]:
    def essentials(p: str):
        return {'id': p,
                'b': getVitalYear(p, 'birth') or getApproxBirth(dict(), p),
                'd': getVitalYear(p, 'death')}

    descent = [essentials(p)]
    spouses = getSpouse(p)
    for spouse in spouses:
        if spouse:
            if people[p]['gender'] == 'F':
                descent.insert(-2, essentials(spouse))
            else:
                descent.append(essentials(spouse))
        for child in getChildren(p, spouse):
            childDescent = generateZegelchart(child)
            descent[-1:-1] = childDescent
    return descent
