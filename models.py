import re
from typing import Dict, List, Literal, Optional, Set
from edtf import (
    EDTFObject,
    Interval,
    Level1Interval,
    Level2Interval,
    UncertainOrApproximate,
    UnspecifiedIntervalSection,
    parse_edtf,
    parser,
    Date as EDTFDate,
)
from time import strftime
import unicodeit


@property
def getPrecision(self) -> str | None:
    date = getattr(self, "date", None)
    return getattr(date, "precision", None)


for dateType in [
    "UncertainOrApproximate",
    "PartialUncertainOrApproximate",
    "PartialUnspecified",
]:
    cls = getattr(parser.parser_classes, dateType, None)
    if cls and not hasattr(cls, "precision"):
        setattr(cls, "precision", getPrecision)


def sanitize(value: str) -> str:
    result = re.sub(r"(?<!\\)([#&$])", r"\\\1", unicodeit.replace(value.strip()))
    return result.replace("\u2212", "-")


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
        self.first = sanitize(first) if first != "---" else first
        self.middle = sanitize(middle)
        self.last = sanitize(last)
        self.shortname = shortname or (self.__str__() if last else self.first)
        self.title = sanitize(title)
        self.antonym = sanitize(antonym)
        self.nickname = sanitize(nickname)

    def __str__(self):
        return f"{self.first} {self.last}".strip()

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def unicode(self):
        return {
            "first": unicodeit.replace(self.first),
            "last": unicodeit.replace(self.last),
        }


class Vitals:
    def __init__(self, date: Optional[str] = None, place: Optional[str] = None):
        self.date = Date(date)
        self.place = sanitize(place) if place else None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getYear(self):
        return self.date.getYear()

    def __str__(self):
        date = self.date.preposition() if self.date else ""
        place = f"in {self.place}" if self.place else ""
        return ", ".join(filter(None, [date, place]))


Children = Set[str]


class Marriage:
    def __init__(
        self,
        date: Optional[str] = None,
        place: Optional[str] = None,
        children: Children = set(),
        adulterous: bool = False,
    ):
        self.date = Date(date)
        self.place = sanitize(place) if place else None
        self.children = set(children)
        self.adulterous = adulterous

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def getYear(self):
        return self.date.getYear()

    def __str__(self):
        date = self.date.preposition() if self.date else ""
        place = f"in {self.place}" if self.place else ""
        return ", ".join(filter(None, [date, place]))


class Date:
    def __init__(self, value: Optional[str]):
        self.raw = str(value) if value is not None else None
        self.edtf: Optional[EDTFObject] = parse_edtf(self.raw) if self.raw else None  # type: ignore

    def getYear(self) -> Optional[int]:
        if not self.edtf:
            return None
        if isinstance(self.edtf, EDTFDate | UncertainOrApproximate):
            return int(self.edtf._strict_date().tm_year)  # type: ignore
        if isinstance(self.edtf, Interval | Level1Interval | Level2Interval):
            if isinstance(self.edtf.lower, UnspecifiedIntervalSection) and isinstance(
                self.edtf.upper, UnspecifiedIntervalSection
            ):
                lower_year = self.edtf.lower._strict_date().tm_year  # type: ignore
                upper_year = self.edtf.upper._strict_date().tm_year  # type: ignore
                return int((lower_year + upper_year) / 2)
            if isinstance(self.edtf.lower, UnspecifiedIntervalSection):
                return int(self.edtf.lower._strict_date().tm_year)  # type: ignore
            if isinstance(self.edtf.upper, UnspecifiedIntervalSection):
                return int(self.edtf.upper._strict_date().tm_year)  # type: ignore
        return None

    def __bool__(self):
        return bool(self.edtf)

    def preposition(self) -> str:
        if not self.edtf:
            return ""
        if isinstance(self.edtf, EDTFDate | UncertainOrApproximate):
            precision = self.edtf.precision  # type: ignore
            preposition = "on" if precision == "day" else "in"
            f = formatPrecision if isinstance(self.edtf, EDTFDate) else formatUncertain
            return f"{preposition} {f(self.edtf)}"  # type: ignore
        if isinstance(self.edtf, Interval):
            f = formatPrecision if isinstance(self.edtf, EDTFDate) else formatUncertain
            return f"between {f(self.edtf.lower)} and {f(self.edtf.upper)}"  # type: ignore
        if isinstance(self.edtf, Level1Interval | Level2Interval):
            fl = (
                formatPrecision
                if isinstance(self.edtf.lower, EDTFDate)
                else formatUncertain
            )
            fu = (
                formatPrecision
                if isinstance(self.edtf.upper, EDTFDate)
                else formatUncertain
            )
            if isinstance(self.edtf.lower, UnspecifiedIntervalSection) and isinstance(
                self.edtf.upper, UnspecifiedIntervalSection
            ):
                return f"between {fl(self.edtf.lower)} and {fu(self.edtf.upper)}"  # type: ignore
            if isinstance(self.edtf.lower, UnspecifiedIntervalSection):
                return f"after {fl(self.edtf.lower)}"  # type: ignore
            if isinstance(self.edtf.upper, UnspecifiedIntervalSection):
                return f"before {fu(self.edtf.upper)}"  # type: ignore
        return self.raw or ""


def formatPrecision(edtf: EDTFDate | UncertainOrApproximate) -> str:
    precision = edtf.precision  # type: ignore
    if precision == "day":
        return strftime("%B %-d, %Y", edtf._strict_date())
    if precision == "month":
        return strftime("%B of %Y", edtf._strict_date())
    else:
        return strftime("%Y", edtf._strict_date())


def formatUncertain(edtf: UncertainOrApproximate) -> str:
    p = []
    if edtf.is_uncertain:
        p.append("approx.")
    if edtf.is_approximate:
        p.append("c.")
    return (" ".join(p) + " " + formatPrecision(edtf)).strip()


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
        self,
        date: Optional[str] = None,
        place: Optional[str] = None,
        cemetery: Optional[str] = None,
        plusCode: Optional[str] = None,
    ):
        self.date = Date(date)
        self.place = sanitize(place) if place else None
        self.cemetery = sanitize(cemetery) if cemetery else None
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
        self.history = sanitize(history)
        self.sources = sources if sources and isinstance(sources, dict) else {}
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


class Cousin:
    def __init__(self, degree: int, removed: int):
        self.degree = degree
        self.removed = removed

    def __str__(self):
        degree = lambda d: {0: "th", 1: "1st", 2: "2nd", 3: "3rd"}.get(d, f"{d}th")
        removed = lambda r: {0: "", 1: "once", 2: "twice"}.get(r, f"{r} times") + (
            " removed" if r else ""
        )
        return f"{degree(self.degree)} cousins {removed(self.removed)}".strip()
