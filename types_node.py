from typing import Dict, List, Literal, Optional, Set, TypedDict


class Name(TypedDict):
    first: str
    middle: Optional[str]
    last: Optional[str]
    shortname: Optional[str]
    title: Optional[str]
    antonym: Optional[str]
    nickname: Optional[str]


class Vitals(TypedDict):
    date: str
    place: str


class Marriage(TypedDict):
    date: str
    place: str
    children: Set[str]


class Person(TypedDict):
    id: str
    name: Name
    gender: Literal["M", "F", "m", "f"]
    birth: Optional[Vitals]
    death: Optional[Vitals]
    marriage: Optional[Dict[str, Marriage]]
    child: Optional[Set[str]]
    spouse: Optional[List[str]]
    father: Optional[str]
    mother: Optional[str]
    generation: Optional[int]
    note: Optional[str]
    history: Optional[str]
    sources: Optional[List[str]]
    tree: Optional[bool]
    army: Optional[bool]
    kia: Optional[bool]
    pow: Optional[bool]
    mason: Optional[bool]
    crusade: Optional[bool]
    templar: Optional[bool]
    lost: Optional[bool]
