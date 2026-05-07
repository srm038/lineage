from typing import Dict, List, Literal, Optional, Set

from edtf import EDTFObject, parse_edtf, text_to_edtf
from pyparsing import ParseResults


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
        self.shortname = shortname or (self.__str__() if last else self.first)
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
        self.date = Date(date)
        self.place = place

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getYear(self):
        return self.date.getYear()


Children = Set[str]


class Marriage:
    def __init__(
        self,
        date: str | int | None = None,
        place: str | None = None,
        children: Children = set(),
    ):
        self.date = Date(date)
        self.place = place
        self.children = set(children)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getYear(self):
        return self.date.getYear()


class Date:
    def __init__(self, value: Optional[str | int]):
        self.value = str(value) if value is not None else None
        edtf = text_to_edtf(self.value) if self.value else None
        self.edtf = parse_edtf(edtf) if edtf else None

    def getYear(self) -> Optional[int]:
        if not self.edtf:
            return None
        return int(self.edtf.year)  # type: ignore

    def __bool__(self):
        return bool(self.value)

    def __str__(self):
        return self.value


class Marriages(Dict[Optional[str], Marriage]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        # Return a safe Marriage() when key missing to avoid KeyErrors in callers
        if key in self:
            return super().__getitem__(key)
        return Marriage()


class Buried:
    def __init__(
        self, date: str = "", place: str = "", cemetery: str = "", plusCode: str = ""
    ):
        self.date = Date(date)
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
        marriage: Optional[Marriages] = None,
        child: Optional[Set[str]] = None,
        spouse: Optional[Set[str]] = None,
        father: Optional[str] = None,
        mother: Optional[str] = None,
        generation: None | int = None,
        note: str = "",
        history: str = "",
        sources: Optional[Dict[str, str]] = None,
        tree: Optional[bool] = None,
        army: Optional[bool] = None,
        kia: Optional[bool] = None,
        pow: Optional[bool] = None,
        mason: Optional[bool] = None,
        crusade: Optional[bool] = None,
        templar: Optional[bool] = None,
        lost: Optional[bool] = None,
        buried: Optional[dict] = None,
        blazon: None | Set[str] | list | str = None,
    ):
        self.id = id
        self.name = name if isinstance(name, Name) else Name(**name)
        self.gender = gender.upper()
        self.birth = Vitals(**birth) if birth else Vitals()
        self.death = Vitals(**death) if death else Vitals()
        self.marriage = marriage or Marriages()
        self.child = set(child) if child is not None else set()
        self.spouse = (
            set(spouse)
            if spouse is not None
            else {k for k in self.marriage.keys() if k}
        )
        self.father = father
        self.mother = mother
        self.generation = generation
        self.note = note
        self.history = history
        self.sources = sources if isinstance(sources, dict) else (sources or {})
        self.tree = tree
        self.army = army
        self.kia = kia
        self.pow = pow
        self.mason = mason
        self.crusade = crusade
        self.templar = templar
        self.lost = lost
        self.buried = Buried(**buried) if buried else Buried()
        if isinstance(blazon, list):
            self.blazon = set(blazon)
        elif blazon:
            self.blazon = set([blazon])
        else:
            self.blazon = set()

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getFecundSpouses(self) -> List[str]:
        return [
            s
            for s in self.marriage.keys()
            if s is not None and self.marriage[s].children
        ]

    def getPronoun(self) -> str:
        return {"M": "He"}.get(self.gender, "She")


class People(Dict[Optional[str], Person]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        # For missing or None keys return a safe stub Person instead of raising
        if key is None:
            return Person(id="null", name=Name(first="", last=""), gender="M")
        if key in self:
            return super().__getitem__(key)
        return Person(id="null", name=Name(first="", last=""), gender="M")
