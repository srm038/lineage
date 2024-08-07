import os
import datetime
import xlrd
import csv
import bs4
import math
from typing import *

people = dict()
generations = list()


def isDateFull(d: str) -> bool:
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


def in_full_tree(p):
    for g in generations:
        if p in people and people[p]['gender'] == 'F':
            if not people[p]['father'] and not people[p]['mother']:
                return False
        if p in g:
            return True
    return False


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


def individual(p):
    # child_check(p)
    person = people[p]
    ancestor = '[p]' if not person['father'] else ''
    name = f"{person['first']} {person['middle'] + ' ' if person['middle'] else ''}{person['last']}"
    name += f"\index{{{person['last']}!{person['first'] + (' ' if person['middle'] else '')}{person['middle']}}}"
    if person['title']:
        name = f"{person['title']} {name}"
    patriline = getLineage(p, 'father')
    if patriline:
        patriline = f" ({patriline})"
    birth, death, accolades, married, spousebirth, spousedeath, sources = '', '', '', '', '', '', ''
    if person['birthdate']:
        full = isDateFull(person['birthdate'])
        birth = f" was born {'on' if full else 'in'} {person['birthdate']}"
        birth += ',' if person['birthplace'] else ''
    if person['birthplace']:
        birth += f" in {person['birthplace']}"
    if any([person['birthdate'], person['birthplace']]):
        birth += ';' if any([person['deathdate'], person['deathplace']]) else '.'
    if person['deathdate']:
        full = isDateFull(person['deathdate'])
        death = f" died {'on' if full else 'in'} {person['deathdate']}"
        death += ',' if person['deathplace'] else ''
    if person['deathplace']:
        death += f" in {person['deathplace']}"
    death += '.' if death else ''
    if not birth and not death:
        birth = '.'
    for a in ['army', 'mason']:
        if person[a]:
            accolades += f"\\{a}"
    if accolades:
        accolades = f" {accolades}"
    if type(person['spouse']) == str:
        person['spouse'] = [person['spouse']]
    for s in sorted(person['spouse'], key=lambda x: get_marriage_year(p, x)):
        spousebirth, spousedeath = '', ''
        spousepatriline = getLineage(s, 'father')
        if spousepatriline:
            spousepatriline = f" ({spousepatriline})"
        if s:
            spouse = f"\\namelinkbold{{{s}}}{{{people[s]['shortname']}}}{spousepatriline}" if in_full_tree(
                s) else f"\\textbf{{{people[s]['shortname']}}}"
            if in_full_tree(s):
                spouse += f"\index{{{people[s]['last']}!{people[s]['first'] + (' ' if people[s]['middle'] else '')}{people[s]['middle']}}}"
        else:
            continue
        ns = person['spouse'].index(s) + 1
        ns = f"({ns}) " if len(person['spouse']) > 1 else ''
        married += f"\n{'He' if person['gender'] == 'M' else 'She'} married {ns}{spouse}"
        if person.get('marriagedate', dict()).get(s):
            married += f" on {person['marriagedate'][s]}"
            married += ',' if person['marriageplace'].get(s) else ''
        if person.get('marriageplace', dict()).get(s):
            married += f" in {person['marriageplace'].get(s)}"
        married += '.'
        if people[s]['birthdate'] or any([people[s]['father'], people[s]['mother']]):
            spousebirth = f"{' She' if person['gender'] == 'M' else 'He'} was born"
        if people[s]['birthdate']:
            full = isDateFull(people[s]['birthdate'])
            spousebirth += f" {'on' if full else 'in'} {people[s]['birthdate']}"
            if people[s]['birthplace']:
                spousebirth += ',' if person['birthplace'] else ''
                spousebirth += f" in {people[s]['birthplace']}"
            spousebirth += ';' if people[s]['deathdate'] or people[s]['deathplace'] else '.'
        if any([people[s]['father'], people[s]['mother']]):
            if spousebirth[-1] in [';', '.']:
                spousebirth = spousebirth[:-1]
            if people[s]['father'] in people:
                spousefather = f"\\namelink{{{people[s]['father']}}}{{{people[people[s]['father']]['shortname']}}}" if in_full_tree(
                    people[s]['father']) else f"{people[people[s]['father']]['shortname']}"
            else:
                spousefather = people[s]['father']
            if people[s]['mother'] in people:
                spousemother = f"\\namelink{{{people[s]['mother']}}}{{{people[people[s]['mother']]['shortname']}}}" if in_full_tree(
                    people[s]['mother']) else f"{people[people[s]['mother']]['shortname']}"
            else:
                spousemother = people[s]['mother']
            spouseparents = ''
            if spousefather:
                spouseparents = f" to {spousefather}"
                if spousemother:
                    spouseparents += f" and {spousemother}"
            elif spousemother:
                spouseparents = f" to {spousemother}"
            spousebirth += spouseparents
            spousebirth += ';' if people[s]['deathdate'] or people[s]['deathplace'] else '.'
        if people[s]['deathdate']:
            full = isDateFull(people[s]['deathdate'])
            spousedeath = ' She' if not people[s]['birthdate'] and not people[s]['birthplace'] else ''
            spousedeath += f" died {'on' if full else 'in'} {people[s]['deathdate']}"
            if people[s]['deathplace']:
                spousedeath += ',' if person['deathplace'] else ''
                spousedeath += f" in {people[s]['deathplace']}"
            spousedeath += '.' if spousedeath else ''
        married = f"{married}{spousebirth}{spousedeath}"

    history = '\n' + person['history'] if person['history'] else ''
    for s in person['spouse']:
        if not s or not people[s]['history']:
            continue
        else:
            history = f" {people[s]['history']}"

    children = ''
    for s in person['children']:
        if not person['children'][s]:
            continue
        if s and s in people:
            children += f"\n\n{person['shortname']} and {people[s]['shortname']}\\children\n\n"
        elif s and s not in people:
            children += f"\n\n{person['shortname']} and {s}\\children\n\n"
        elif not s:
            children += f"\n\n{person['shortname']}\\children\n\n"
        for c in person['children'][s]:
            if c not in people:
                print(f"{c} doesn't have an entry")
                continue
        for c in sorted([d for d in person['children'][s] if d in people], key=lambda y: get_birth_year(y)):
            main = '[+]' if c in person['child'] else ''
            born = f"born {people[c]['birthdate']}." if people[c]['birthdate'] else '.'
            if people[c]['gender'] == 'F' and people[c].get('generation'):
                if type(people[c]['spouse']) == str:
                    born += f" She married \\namelink{{{people[c]['spouse']}}}{{{people[people[c]['spouse']]['shortname']}}}."
                else:
                    for cs in people[c]['spouse']:
                        if not people.get(cs, dict()).get('generation'):
                            continue
                        born += f" She married \\namelink{{{cs}}}{{{people[cs]['shortname']}}}."
            children += f"\\childlist{main}{{{c if main else ''}}}{{{people[c]['title'] + ' ' if people[c]['title'] else ''}{people[c]['shortname']}}}{{{born}}}\n"
        if children:
            children = children[:-1]

    buried = ''
    if person['buried']:
        buried = f"\n\n\\buried \\href{{{person['buriedlink']}}}{{{person['buried']}}}"

    if person['sources']:
        # sources = '\\begin{footnoterange}'
        # for s in person['sources']:
        #     sources += f"\\footnote{{{s}}}"
        # sources += '\\end{footnoterange}'
        sources += '\n\\begin{source}'
        all_sources = person['sources']
        for s in person['spouse']:
            if not s:
                continue
            all_sources |= people[s].get('sources', set())
        for s in sorted(all_sources):
            sources += f"\\item{{{s}}}"
        sources += '\\end{source}'

    return f"\individual{ancestor}{{{p}}}{{{name}}}{accolades}{patriline}{birth}{death}{married}{history}{buried}{children}{sources}"


def descent(p, p0):
    d = [[p]]
    while d[0][-1] != p0:
        children = list(people[d[0][-1]]['child'])
        if len(children) > 1:
            # print(children)
            d.append(d[-1] + descent(children[1], p0)[0])
        d[0].append(children[0])
        # print(d)

    def men(m):
        i = 0 if people[m[0]]['gender'] == 'M' else 1
        c = 0
        for j in m[i:]:
            if people[j]['gender'] == 'M':
                c += 1
            else:
                break
        return c

    return sorted(d, key=lambda m: men(m[::-1]), reverse=True)


def generation_groups(p0):
    generations.clear()
    for p in people:
        try:
            people[p]['generation'] = max(len(d) - 1 for d in descent(p, p0))
        except:
            pass
    max_g = max(people[p].get('generation', 0) for p in people)
    for g in range(max_g + 1):
        generations.insert(0, {p for p in people if 'generation' in people[p] and people[p]['generation'] == g})


def ancestors(p):
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


def generation_groups_old(p0):
    # TODO multiple lines of descent
    generations.clear()
    generations.append({p0})
    done = set({p0})
    to_search = set([people[p0]['father'], people[p0]['mother']])
    while to_search:
        generations.insert(0, set())
        for p in list(to_search):
            if p in done:
                to_search.remove(p)
                continue
            # if any(s and s in people[p].get('spouse') for s in done):
            #     continue
            if p and in_full_tree(p):
                generations[0].add(p)
                to_search |= set([people[p]['father'], people[p]['mother']])
                to_search.remove(p)
                done.add(p)
            else:
                to_search.remove(p)
                done.add(p)
        print(to_search, generations)
    if not generations[0]:
        del generations[0]


def get_birth_year(p):
    try:
        date = people[p]['birthdate'].split(' ')
    except:
        return int(people[p]['birthdate'])
    if not date:
        year = 0
    if len(date) == 3:
        year = int(date[2])
    elif len(date) == 2 and int(date[1]) > 31:
        year = int(date[1])
    else:
        year = 0
    return year


def get_death_year(p):
    try:
        date = people[p]['deathdate'].split(' ')
    except:
        return int(people[p]['deathdate'])
    if not date:
        year = 0
    if len(date) == 3:
        year = int(date[2])
    elif len(date) == 2 and int(date[1]) > 31:
        year = int(date[1])
    else:
        year = 0
    return year


def get_marriage_year(p, s):
    if not s:
        return 0
    if s not in people[p]['marriagedate']:
        return 0
    try:
        date = people[p]['marriagedate'][s].split(' ')
    except:
        return int(people[p]['marriagedate'][s])
    if not date:
        year = 0
    if len(date) == 3:
        year = int(date[2])
    elif len(date) == 2 and int(date[1]) > 31:
        year = int(date[1])
    else:
        year = 0
    return year


def export(file, p0):
    import_people(file)
    generation_groups(p0)
    with open(fr"{os.getcwd()}\{file}_generated.tex", 'w') as f:
        f.write(f"\\chapter*{{{file}}}\n\n")
        for g in generations:
            if p0 in g and people[p0]['gender'] == 'F':
                continue
            f.write(f"\\generationgroup\n\n")
            for p in sorted(list(g), key=lambda y: get_birth_year(y)):
                if people[p]['gender'] == 'F' and 'spouse' in people[p] and people[p]['spouse'][0]:
                    continue
                try:
                    f.write(f"{individual(p)}\n\n")
                except:
                    print(f"{p} error")
    h_tree(file, p0)
    line_chart(file, p0)
    follow(lost=True)
    unsourced()
    print(f"{len(set.union(*generations))} total ancestors")


def follow(g=None, lost=False):
    i = 0
    for p in sorted(people, key=lambda p: get_birth_year(p)):
        if lost and people[p]['lost']:
            continue
        if g and people[p].get('generation') != g:
            continue
        if people[p]['gender'] == 'F' and people[p].get('spouse') and people[p].get('generation'):
            if not people[p]['father']:
                print(p, people[p]['spouse'], get_birth_year(p), people[p]['note'], people[people[p]['spouse']]['note'])
                i += 1
            elif not in_full_tree(people[p]['father']):
                print(p, people[p]['spouse'], get_birth_year(p), people[p]['note'], people[people[p]['spouse']]['note'])
                i += 1
        if people[p]['gender'] == 'M' and people[p].get('generation') and not in_full_tree(people[p]['father']):
            print(p, get_birth_year(p), people[p]['note'])
            i += 1
    print(f"{i} threads to pull")


def unsourced(g=None):
    i = 0
    for g in generations[::-1][2:]:
        for p in g:
            if people[p]['gender'] == 'M' and not people[p]['sources']:
                print(p, people[p]['shortname'], get_birth_year(p), people[p]['note'])
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
            b = get_birth_year(p)
            d = get_death_year(p)
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
        if not get_death_year(p):
            if get_birth_year(p) and currentYear - get_birth_year(p) < 100:
                done[p]['d'] = currentYear
                continue
            unknownd.add(p)
            done[p]['d'] = max(done[c]['b'] for c in people[p]['child'])
            done[p]['d'] = max(done[p]['d'], max([get_marriage_year(p, s) for s in people[p]['marriagedate']] or [0]))
        if not get_birth_year(p):
            unknownb.add(p)
            done[p]['b'] = min([
                                   done.get(people[p].get('mother', ''), dict()).get('d', 3000) or 3000,
                                   done.get(people[p].get('father', ''), dict()).get('d', 3000) or 3000,
                                   min(done[c]['b'] for c in people[p]['child']) - 20
                               ] + (
                                   [get_marriage_year(p, s) or 3000 for s in people[p]['spouse'] if s in people] if
                                   people[p]['gender'] == 'M' else
                                   [get_marriage_year(p, people[p]['spouse']) or 3000]
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
            if not get_marriage_year(p, s):
                continue
            flags += f"<image x='{get_marriage_year(p, s) * w / widx:.3f}' y='{done[p]['y'] - 4.5 * 72 / 96:.3f}' " \
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
        if in_full_tree(p) and month in str(people[p]['birthdate']):
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


def living_ancestors(p):
    b0 = get_birth_year(p)
    d0 = get_death_year(p)
    alive = set()
    if not b0 and not d0:
        return n
    for a in ancestors(p):
        b = get_birth_year(a)
        d = get_death_year(a)
        if not d and 2021 - b < 100:
            d = 2021
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
    for p in sorted(byAge, key=lambda p: get_death_year(p) - get_birth_year(p)):
        print(people[p]['shortname'], get_death_year(p) - get_birth_year(p))
