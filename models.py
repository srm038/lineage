from typing import Dict, List, Literal, Optional, Set


class Name:
    def __init__(
        self,
        first: str,
        middle: str = "",
        last: str = "",
        shortname: str = "",
        title: str = "",
        antonym: str = "",
        nickname: str = "",
    ):
        self.first = first.strip().replace(":", '\\"')
        self.middle = middle.strip().replace(":", '\\"')
        self.last = last.strip().replace(":", '\\"')
        self.shortname = (
            shortname
            if shortname
            else f"{self.first} {self.last}" if last else self.first
        )
        self.title = title.strip().replace(":", '\\"')
        self.antonym = antonym.strip().replace(":", '\\"')
        self.nickname = nickname.strip().replace(":", '\\"')

    def __str__(self):
        return f"{self.first} {self.last}"

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Vitals:
    def __init__(self, date: str | int | None = None, place: str | None = None):
        self.date = date
        self.place = place

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getYear(self):
        if not self.date:
            return None
        if isinstance(self.date, int):
            return self.date
        return int(self.date.split(" ")[-1])


Children = Set[str]


class Marriage:
    def __init__(
        self,
        date: str | int | None = None,
        place: str | None = None,
        children: Children = set(),
    ):
        self.date = date
        self.place = place
        self.children = set(children)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getYear(self):
        if not self.date:
            return None
        if isinstance(self.date, int):
            return self.date
        return int(self.date.split(" ")[-1])


class Buried:
    def __init__(
        self, date: str = "", place: str = "", cemetery: str = "", plusCode: str = ""
    ):
        self.date = date
        self.place = place
        self.cemetery = cemetery
        self.plusCode = plusCode

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Person:
    def __init__(
        self,
        id: str,
        name: Name,
        gender: Literal["M", "F", "m", "f"],
        birth: Optional[Vitals] = None,
        death: Optional[Vitals] = None,
        marriage: Dict[str, Marriage] = {},
        child: Set[str] = set(),
        spouse: Set[str] = set(),
        father: Optional[str] = None,
        mother: Optional[str] = None,
        generation: None | int = None,
        note: str = "",
        history: str = "",
        sources: Dict[str, str] = {},
        tree: Optional[bool] = None,
        army: Optional[bool] = None,
        kia: Optional[bool] = None,
        pow: Optional[bool] = None,
        mason: Optional[bool] = None,
        crusade: Optional[bool] = None,
        templar: Optional[bool] = None,
        lost: Optional[bool] = None,
        buried: Optional[Buried] = None,
        blazon: None | Set[str] = None,
    ):
        self.id = id
        self.name = Name(**name)
        self.gender = gender.upper()
        self.birth = Vitals(**birth) if birth else Vitals()
        self.death = Vitals(**death) if death else Vitals()
        self.marriage = {k: Marriage(**v) for k, v in marriage.items()}
        self.child = set(child)
        self.spouse = spouse if spouse else {k for k in marriage.keys() if k}
        self.father = father
        self.mother = mother
        self.generation = generation
        self.note = note
        self.history = history
        self.sources = sources if isinstance(sources, dict) else {s: s for s in sources}
        self.tree = tree
        self.army = army
        self.kia = kia
        self.pow = pow
        self.mason = mason
        self.crusade = crusade
        self.templar = templar
        self.lost = lost
        self.buried = Buried(**buried) if buried else Buried()
        self.blazon = (
            set(blazon)
            if type(blazon) == list
            else (set([blazon]) if blazon else set())
        )

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getFecundSpouses(self) -> List[str]:
        return [s for s in self.marriage.keys() if self.marriage[s].children]

    def getPronoun(self) -> str:
        return {"M": "He"}.get(self.gender, "She")
