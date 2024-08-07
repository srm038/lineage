import os
import datetime
import xlrd
import csv
import bs4
import math
from typing import *
import warnings

people = dict()
generations = list()


def isDateFull(d: Union[str, int]) -> bool:
    d = str(d)
    d = d.split(' ')
    return len(d) == 3


def import_people(file):
    people.clear()
    wb = xlrd.open_workbook(f"{file}.xlsx")
    sh = wb.sheet_by_index(0)
    with open(fr"{os.getcwd()}\{file}.txt", 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter='\t', lineterminator='\n')
        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))
    with open(fr"{os.getcwd()}\{file}.txt", 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split('\t')
        for line in lines[1:]:
            data = line.split('\t')
            if data[0] in people:
                print(f"{data[0]} duplication error")
                continue
            people.update({data[0]: dict(zip(headers[1:], data[1:]))})
            for h in headers:
                if h in people[data[0]]:
                    people[data[0]][h] = people[data[0]][h].strip()
            people[data[0]]['gender'] = people[data[0]]['gender'].upper()
            while type(people[data[0]]['marriagedate']) == str:
                people[data[0]]['marriagedate'] = eval(people[data[0]]['marriagedate']) if people[data[0]][
                    'marriagedate'] else dict()
            while type(people[data[0]]['marriageplace']) == str:
                people[data[0]]['marriageplace'] = eval(people[data[0]]['marriageplace']) if people[data[0]][
                    'marriageplace'] else dict()
            while type(people[data[0]]['children']) == str:
                people[data[0]]['children'] = eval(people[data[0]]['children']) if people[data[0]][
                    'children'] else dict()
            while type(people[data[0]]['child']) == str:
                people[data[0]]['child'] = eval(people[data[0]]['child']) if people[data[0]]['child'] else set()
            while type(people[data[0]]['sources']) == str:
                people[data[0]]['sources'] = eval(people[data[0]]['sources']) if people[data[0]]['sources'] else set()
            people[data[0]]['shortname'] = people[data[0]]['shortname'].replace(':', "\\\"")
            people[data[0]]['shortname'] = people[data[0]]['shortname'].strip()
            people[data[0]]['last'] = people[data[0]]['last'].replace(':', "\\\"")
            for i in ['birthdate', 'birthplace', 'deathdate', 'deathplace', 'buried',
                      'army', 'mason', 'history', 'pow']:
                try:
                    people[data[0]][i] = eval(people[data[0]][i]) if people[data[0]][i] else ''
                except SyntaxError:
                    pass
                except NameError:
                    pass
                except KeyError:
                    pass
            if people[data[0]]['children']:
                people[data[0]]['spouse'] = list(people[data[0]]['children'])
            # print(data[0])
            people[data[0]]['lost'] = bool(float(people[data[0]]['lost'] or 0))
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
        for p in people:
            if people[p]['gender'] == 'M':
                for s in people[p].get('children', dict()):
                    for c in people[p]['children'][s]:
                        if not c:
                            continue
                        people[c]['father'] = p
                        people[c]['mother'] = s
            if people[p]['gender'] == 'F':
                for c in people[p].get('child', set()):
                    people[c]['mother'] = p


def inFullTree(p: str, p0: str) -> bool:
    return p in getAncestors(p0)


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
    patriline = printLineage(p, 'father')
    birth = combineDatePlace(person, 'birth')
    death = combineDatePlace(person, 'death')
    vitals = combineVitals(birth, death)
    accolades = getAccolades(person)
    spouseDetails = generateSpouse(person, p0)
    history = person.get('history', '')
    childrenDetails = getChildrenDetails(person, p0)
    burialDetails = getBurialDetails(person)
    sources = getSources(person)

    return buildParagraphs(
        buildParagraph(
            buildSentence(
                f"\individual{ancestor}{{{p}}}{{{buildSentence(title, name)}}}{nameIndex}}}",
                accolades,
                patriline,
                vitals
            ),
            *spouseDetails
        ),
        childrenDetails,
        burialDetails,
        sources
    )


def getSources(person: Dict) -> str:
    if not person['sources']:
        return ''
    allSources = person['sources']
    for s in person['spouse']:
        if not s:
            continue
        allSources |= people[s].get('sources', set())
    sources = [f"\\item{{{s}}}" for s in sorted(allSources)]
    return buildSentence('\\begin{source}', *sources, '\\end{source}')


def getBurialDetails(person: Dict) -> str:
    if person['buried']:
        return f"\\buried \\href{{{person['buriedlink']}}}{{{person['buried']}}}"
    return ''


def getChildrenDetails(person: Dict, p0: str) -> str:
    childrens = []
    for s in person['children']:
        if not person['children'][s]:
            warnings.warn(f"{c} doesn't have an entry", Warning)
            continue
        parentDetails = getParentDetails(person, s) + '\n'
        children = []
        for c in sorted(person['children'][s], key=lambda y: getVitalYear(y, 'birth')):
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
    birth = childBirth(c)
    marriage = childMarriage(c, mainLine, p0)
    return f"\\childlist{mainLine}{{{c if mainLine else ''}}}" \
           f"{{{buildSentence(title, people[c]['shortname'])}}}" \
           f"{{{joinComma(birth, marriage)}}}"


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
    if people[c]['birthdate']:
        return f"born {people[c]['birthdate']}"
    return ''


def getMainLine(person: Dict, c: str) -> str:
    if c in person['child']:
        return '[+]'
    return ''


def getParentDetails(person: Dict, s: str) -> str:
    if not s:
        return f"{person['shortname']}\\children"
    if s not in people:
        return f"{person['shortname']} and {s}\\children"
    return f"{person['shortname']} and {people[s]['shortname']}\\children"


def generateSpouse(person: Dict, p0: str):
    spouse: list = person['spouse']
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
    vitalDate: str = person[vitalDateKey]
    vitalPlace: str = person[vitalPlaceKey]
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
    vitalDate = person[vitalDateKey].get(s)
    vitalPlace = person[vitalPlaceKey].get(s)
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


def getNameIndex(person):
    firstName = person['first']
    middleName = person['middle']
    lastName = person['last']
    nameIndex = f"\index{{{lastName}!{joinName(firstName, middleName)}}}"
    return nameIndex


def getFullName(person: Dict) -> str:
    firstName = person['first']
    middleName = person['middle']
    lastName = person['last']
    name = joinName(firstName, middleName, lastName)
    return name


def joinName(*name: iter) -> str:
    joinedName = ' '.join(filter(None, name))
    return joinedName


def getAncestorTag(person: Dict) -> str:
    if not person['father']:
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


def getVitalYear(p: str, vital: str) -> int:
    if vital not in ['birth', 'death']:
        raise KeyError(f"{vital} is not a vital statistic")
    date: Union[int, str] = people.get(p, {}).get(f'{vital}date', 0)
    if not date:
        return 0
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


def export(file, p0):
    import_people(file)
    updateGenerationGroups(p0)
    with open(fr"{os.getcwd()}\{file}_generated.tex", 'w') as f:
        f.write(f"\\chapter*{{{file}}}\n\n")
        for g in generations:
            if p0 in g and people[p0]['gender'] == 'F':
                continue
            f.write(f"\\generationgroup\n\n")
            for p in sorted(list(g), key=lambda y: getVitalYear(y, 'birth')):
                if people[p]['gender'] == 'F' and 'spouse' in people[p] and people[p]['spouse'][0]:
                    continue
                try:
                    f.write(f"{printIndividualEntry(p, p0)}\n\n")
                except:
                    print(f"{p} error")
    # h_tree(file, p0)
    # line_chart(file, p0)
    # follow(lost=True)
    # unsourced()
    print(f"{len(set.union(*generations))} total ancestors")


def follow(g=None, lost=False):
    i = 0
    for p in sorted(people, key=lambda p: getVitalYear(p, 'birth')):
        if lost and people[p]['lost']:
            continue
        if g and people[p].get('generation') != g:
            continue
        if people[p]['gender'] == 'F' and people[p].get('spouse') and people[p].get('generation'):
            if not people[p]['father']:
                print(p, people[p]['spouse'], getVitalYear(p, 'birth'), people[p]['note'],
                      people[people[p]['spouse']]['note'])
                i += 1
            elif not inFullTree(people[p]['father'], p0):
                print(p, people[p]['spouse'], getVitalYear(p, 'birth'), people[p]['note'],
                      people[people[p]['spouse']]['note'])
                i += 1
        if people[p]['gender'] == 'M' and people[p].get('generation') and not inFullTree(people[p]['father'], p0):
            print(p, getVitalYear(p, 'birth'), people[p]['note'])
            i += 1
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


def h_tree(file, p, s=90):
    with open(f"{file}-H.svg", 'r') as f:
        svg = bs4.BeautifulSoup(f, 'xml')

    tree_style = 'fill:white;'
    name_style = 'fill:black;text-anchor:middle;alignment-baseline:middle;'
    duplicate = 'opacity:0.5;'
    lines_style = 'stroke:black;stroke-width:5px;'

    def max_gens(p):

        n = list()
        for c in people:
            try:
                for d in descent(c, p):
                    n.append(d)
            except:
                pass
        return max(len(i) for i in n)

    N = len(generations) // 2

    def position(d, s=s + 15, N=N):

        x = 0
        y = 0
        if len(d) == 1:
            return x, y
        for i, j in enumerate(d[::-1][1:]):
            if not (i + 1) % 2:
                y += (-1 if people[j]['gender'] == 'M' else 1) * s * 2 ** ((N - people[j]['generation'] // 2))
                # y += (-1 if people[j]['gender'] == 'M' else 1) * s * 2 ** ((N - max_gens(j) // 2))
            else:
                x += (-1 if people[j]['gender'] == 'M' else 1) * s * 2 ** ((N - people[j]['generation'] // 2 - 1))
                # x += (-1 if people[j]['gender'] == 'M' else 1) * s * 2 ** ((N - max_gens(j) // 2))
            # print(i, j, x, y, people[j]['generation'])
        return x, y

    tree = ''
    names = ''
    lines = ''
    minx, miny, maxx, maxy = [0] * 4
    # N = 5 // 2

    done = set()
    current = {p}
    while current:
        for c in list(current):
            for d in descent(c, p):
                # print(d, position(d))
                x, y = position(d, N=N)
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
                xc, yc = position(d[1:], N=N)
                tree += f"<rect x='{x - s / 2}' y='{y - s / 2}' width='{s}' height='{s}' style='{tree_style}' />"
                names += f"<text style='{name_style}{duplicate if c in done else ''}'><tspan x='{x}' y='{y}' dy='-2.5pt'>{people[c]['first']}</tspan><tspan x='{x}' y='{y}' dy='7.5pt'>{people[c]['last']}</tspan></text>"
                lines += f"<path d='M {x},{y} L {xc},{yc}' style='{lines_style}'/>"
                done.add(c)
            if people[c]['father'] and people[c]['father'] in people:
                current.add(people[c]['father'])
            if people[c]['mother'] and people[c]['mother'] in people:
                current.add(people[c]['mother'])
            current.discard(c)

    svg.find('svg').clear()
    svg.find('svg').contents = bs4.BeautifulSoup(lines, 'html.parser').contents
    svg.find('svg').contents += bs4.BeautifulSoup(tree, 'html.parser').contents
    svg.find('svg').contents += bs4.BeautifulSoup(names, 'html.parser').contents
    svg.find('svg').attrs.update({'viewBox': f"{minx - s / 2} {miny - s / 2} {maxx - minx + s} {maxy - miny + s}"})

    with open(f"{file}-H.svg", 'w') as f:
        f.write(svg.prettify())


def line_chart(file, root):
    currentYear = datetime.datetime.now().year + 10
    current = {root}
    done = dict()
    while current:
        for p in list(current):
            b = getVitalYear(p, 'birth')
            d = getVitalYear(p, 'death')
            # y = 0 if p == root else 2 * (done[list(people[p]['child'])[0]]['y'] - (1 if people[p]['gender'] == 'F' else -1))
            y = 0
            done.update({p: {'b': b, 'd': d, 'y': y}})
            if people[p]['father'] in people:
                current.add(people[p]['father'])
            if people[p]['mother'] in people:
                current.add(people[p]['mother'])
            current.discard(p)

    unknownb = set()
    unknownd = set()
    for p in done:
        if not getVitalYear(p, 'death'):
            if getVitalYear(p, 'birth') and currentYear - getVitalYear(p, 'birth') < 100:
                done[p]['d'] = currentYear
                continue
            unknownd.add(p)
            done[p]['d'] = max(done[c]['b'] for c in people[p]['child'])
            done[p]['d'] = max(done[p]['d'], max([getMarriageYear(p, s) for s in people[p]['marriagedate']] or [0]))
        if not getVitalYear(p, 'birth'):
            unknownb.add(p)
            done[p]['b'] = min([
                                   done.get(people[p].get('mother', ''), dict()).get('d', 3000) or 3000,
                                   done.get(people[p].get('father', ''), dict()).get('d', 3000) or 3000,
                                   min(done[c]['b'] for c in people[p]['child']) - 20
                               ] + (
                                   [getMarriageYear(p, s) or 3000 for s in people[p]['spouse'] if s in people] if
                                   people[p]['gender'] == 'M' else
                                   [getMarriageYear(p, people[p]['spouse']) or 3000]
                               ))

    N = len(generations)

    def position(d, N=N, s=1):

        y = 0
        if len(d) == 1:
            return 0
        for i, j in enumerate(d[::-1][1:]):
            yy = (1 if people[j]['gender'] == 'F' else -1) * s * 2 ** ((N - people[j]['generation']))
            y += yy
            # print(j, yy)
        return y

    for p in done:
        done[p]['y'] = position(descent(p, root)[0], s=10 * 72 / 96)

    by_pos = sorted([p for p in done if 0 <= done[p]['y']], key=lambda p: done[p]['y'], reverse=False)
    for i, p in enumerate(by_pos):
        if i == 0:
            continue
        done[by_pos[i]]['y'] = done[by_pos[i - 1]]['y'] + 2 * 10 * 72 / 96
    by_pos = sorted([p for p in done if done[p]['y'] <= 0], key=lambda p: done[p]['y'], reverse=True)
    for i, p in enumerate(by_pos):
        if i == 0:
            continue
        done[by_pos[i]]['y'] = done[by_pos[i - 1]]['y'] - 2 * 10 * 72 / 96

    minx = min(done[p]['b'] for p in done) - 10
    maxx = currentYear
    widx = maxx - minx
    w = 18 * 96
    miny = min(done[p]['y'] for p in done) - 10
    maxy = max(done[p]['y'] for p in done) + 10

    with open(f"{file}-lines.svg", 'r') as f:
        svg = bs4.BeautifulSoup(f, 'xml')

    lines = ''
    begats = ''
    names = ''
    years = ''
    flags = ''
    for p in done:
        if people[p].get('mother') in done:
            m = people[p]['mother']
            begats += f"<path d='M {done[p]['b'] * w / widx:.3f},{done[p]['y'] - 5 * 72 / 96:.3f} " \
                      f"V {done[m]['y'] + 10 * 72 / 96:.3f}' " \
                      f"class='{'unbegat' if p in unknownb | unknownd else 'begat'}' id='{m}-{p}'/>"
        if people[p].get('father') in done:
            f = people[p]['father']
            begats += f"<path d='M {done[p]['b'] * w / widx:.3f},{done[p]['y'] + 10 * 72 / 96:.3f} " \
                      f"V {done[f]['y'] - 5 * 72 / 96:.3f}' " \
                      f"class='{'unbegat' if p in unknownb | unknownd else 'begat'}' id='{f}-{p}'/>"
    for p in done:
        # lines += f"<path d='M {done[p]['b'] * w / widx},{done[p]['y']} H {done[p]['d'] * w / widx}' class='{'unknown' if p in unknown else 'known'}' id='{p}'/>"
        if p in unknownb and p not in unknownd:
            p_class = 'unknownb'
            if get_state(p, 'death'):
                flags += f"<image x='{(done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{get_state(p, 'death').lower()}.png'/>"
        elif p not in unknownb and p in unknownd:
            p_class = 'unknownd'
            if get_state(p, 'birth'):
                flags += f"<image x='{(done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{get_state(p, 'birth').lower()}.png' style='transform: translateX(-15.72pt)'/>"
        elif p in unknownb and p in unknownd:
            p_class = 'unknownbd'
        else:
            p_class = 'known'
            if get_state(p, 'death'):
                flags += f"<image x='{(done[p]['d'] + 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{get_state(p, 'death').lower()}.png'/>"
            if get_state(p, 'birth'):
                flags += f"<image x='{(done[p]['b'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                         f"height='9pt' href='flags/{get_state(p, 'birth').lower()}.png' style='transform: translateX(-15.72pt)'/>"
        lines += f"<rect x='{done[p]['b'] * w / widx:.3f}' y='{done[p]['y'] - 5 * 72 / 96:.3f}' " \
                 f"width='{(done[p]['d'] - done[p]['b']) * w / widx:.3f}' height='10pt' id='{p}'" \
                 f"class='{p_class}' />"
        if p in unknownb:
            lines += f"<rect x='{(done[p]['b'] - 5) * w / widx:.3f}' y='{done[p]['y'] - 5 * 72 / 96:.3f}' " \
                     f"width='{6 * w / widx:.3f}' height='10pt' id='{p}-b'" \
                     f"class='unknownbb' />"
        if p in unknownd:
            lines += f"<rect x='{(done[p]['d'] - 1) * w / widx:.3f}' y='{done[p]['y'] - 5 * 72 / 96:.3f}' " \
                     f"width='{6 * w / widx:.3f}' height='10pt' id='{p}-d'" \
                     f"class='unknownda' />"
        for s in people[p].get('marriageplace', []):
            if not getMarriageYear(p, s):
                continue
            flags += f"<image x='{getMarriageYear(p, s) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
                     f"height='9pt' href='flags/{get_state(p, 'marriage')[s].lower()}.png' style='transform: translateX(-7.36pt)'/>"
    for p in done:
        names += f"<text class='name {'nameunknown' if p in unknownb | unknownd else ''}'><tspan dy='2.5pt' x={done[p]['b'] * w / widx:.3f} y={done[p]['y']:.3f}>{people[p]['first']} {people[p]['last']}</tspan></text>"

    for y in range(minx, maxx + 1):
        if not y % 100:
            years += f"<path d='M {y * w / widx:.0f},{maxy} V {miny}' class='year' id='{y}'/>"

    minx = min((done[p]['b'] - 10) * w / widx for p in done)
    maxx = currentYear * w / widx

    svg.find('svg').find('g').clear()
    svg.find('svg').find('g').contents = bs4.BeautifulSoup(years + begats + lines + flags + names,
                                                           'html.parser').contents
    svg.find('svg').attrs.update({'viewBox': f"{minx - 10} {miny - 10} {maxx - minx + 20} {maxy - miny + 20}"})
    svg.find('svg').attrs.update({'height': f"{maxy - miny}", 'width': f"{maxx - minx}"})
    svg.find('style').string = ".known {fill:#888;} " \
                               ".unknownb {fill:url(#ub);} " \
                               ".unknownd {fill:url(#ud);} " \
                               ".unknownbd {fill:url(#ubd);} " \
                               ".unknownda {fill:url(#uda);} " \
                               ".unknownbb {fill:url(#ubb);} " \
                               ".begat {stroke-width:1pt; stroke:#888} " \
                               ".unbegat {stroke-width:1pt; stroke:#aaa} " \
                               ".name {dominant-baseline:middle; paint-order:stroke fill; font: bold 9pt Carlito; fill:black; stroke: white;stroke-width:2px} " \
                               ".nameunknown {fill:#333}" \
                               ".year {stroke-dasharray:5pt; stroke-width:2pt; stroke:#ccc}"
    with open(f"{file}-lines.svg", 'w') as f:
        f.write(svg.prettify())


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


def get_state(p, time):
    if not people[p][f'{time}place']:
        return None
    if time == 'marriage':
        state = {m: people[p][f'{time}place'][m].split(',')[-1].strip() for m in people[p][f'{time}place']}
        return state
    state = people[p][f'{time}place'].split(',')
    return state[-1].strip()


def ancestors_by_age():
    byAge = [p for p in set.union(*generations) if people[p]['birthdate'] and people[p]['deathdate']]
    for p in sorted(byAge, key=lambda p: getVitalYear(p, 'death') - getVitalYear(p, 'birth')):
        print(people[p]['shortname'], getVitalYear(p, 'death') - getVitalYear(p, 'birth'))
