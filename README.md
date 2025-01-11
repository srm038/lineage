# Lineage

This began as a personal project for my own genealogical research. I was not fully satisfied with existing solutions (although they are all commendable in their own way). I wanted two things:

- A more data-exposed approach (human-readable `JSON`)
- Customizable outputs - TeX to PDF, full charts and trees, _etc_.

## Numbering

I chose to use a modified form of the Multiple-Surnames System with the NGQS numbering system. A `+` next to the name of a child indicates that the line of descent passes through them. Multiple family lines are represented, and so each name in the genealogy is assigned a number in ascending order (for simplicity, only descendants who are carried forward are assigned a number). These numbers are provided for names which appear elsewhere in the narrative, particularly in the case of wives and in-laws, to facilitate look-up between similar names. The families are divided by generation, counting down from the earliest known ancestor of any line. To aid in navigation, the patrilinial line is provided for each individual in their own entry.

## Shorthand

Sometimes during research, you need a placeholder for a spouse or child and don't need to create a separate object at the moment. The important information we might need to know is the gender, first name (last name can be inferred from the father for a child), optional last name (for an undetailed spouse), and optional birth year. This information is probably not enough to guarantee uniqueness if `birthYear` is missing, and even then there is a risk of collision. For this reason, we can pass the name into the ID generator which will append a sequence number if the string is non-unique.

```
{gender}|{firstName}|{lastName}|{birthYear}
```

Example:

```
m|john||1845
```

Assuming John's father's last name is Smith, this would be parsed as:

```json
{
  "id": "johnsmith",
  "name": {
    "first": "John",
    "last": "Smith"
  },
  "birth": {
    "date": 1845
  }
}
```
