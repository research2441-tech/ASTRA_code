"""Builds the manuscript: python make_paper.py out.docx"""
import os
import json, sys, re
from builder import Doc
from sec_intro import intro
from sec_related import related
from sec_method import methodology
from sec_data import data_protocol
from sec_results import results, front, discussion, limitations, conclusion, backmatter

W = os.environ.get("ASTRA_WORK", os.path.dirname(os.path.abspath(__file__)))


def ieee(r):
    a = r["authors"].strip().rstrip(".")
    t = r["title"].strip().rstrip(".")
    v = r.get("venue", "").strip()
    vol, iss, pg, yr = r.get("volume", ""), r.get("issue", ""), r.get("pages", ""), r.get("year", "")
    doi, url, typ = r.get("doi", ""), r.get("url", ""), r.get("type", "journal")
    pg = str(pg).replace("--", "-").replace("-", "–") if pg else ""
    if typ == "legal":
        s = f"{t}, *{v}*" + (f", vol. {vol}" if vol else "") + (f", pp. {pg}" if pg else "") + f", {yr}."
        return s + (f" [Online]. Available: {url}" if url else "")
    if typ == "book":
        s = f"{a}, *{t}*. {v + ', ' if v else ''}{yr}."
    elif typ == "preprint":
        s = f'{a}, "{t}," *{v or "arXiv preprint"}*, {yr}.'
    elif typ == "report":
        s = f'{a}, "{t}," {v}, {yr}.'
    elif typ == "conference":
        s = f'{a}, "{t}," in *{v}*, {yr}' + (f", pp. {pg}" if pg else "") + "."
    else:
        s = f'{a}, "{t}," *{v}*' + (f", vol. {vol}" if vol else "") + (f", no. {iss}" if iss else "") + \
            (f", pp. {pg}" if pg and "–" in pg else (f", Art. no. {pg}" if pg else "")) + f", {yr}."
    if doi:
        s += f" doi: {doi}."
    elif url and typ != "conference" and "gesis" not in url:
        s += f" [Online]. Available: {url}"
    return s


def build(labels=None):
    R = json.load(open(f"{W}/paper_numbers.json"))
    refs = {r["key"]: ieee(r) for r in json.load(open("refs_all.json"))}
    d = Doc()
    if labels:
        d.labels = dict(labels)
    front(d, R)
    intro(d, R)
    related(d, R)
    methodology(d, R)
    data_protocol(d, R)
    results(d, R)
    discussion(d, R)
    limitations(d, R)
    conclusion(d, R)
    backmatter(d, R)
    d.h1("REFERENCES", numbered=False)
    d.references(refs)
    return d


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ASTRA_manuscript.docx"
    d1 = build()
    d2 = build(d1.labels)
    assert d2.labels == d1.labels
    d2.save(out)
    txt = "".join(d2.body)
    print("display equations:", d2.n_display, "inline math:", d2.n_inline, "figures:", d2.fig, "tables:", d2.tab,
          "refs:", len(d2.cite_order), "unresolved refs:", re.findall(r"\?\?[\w:]+", txt)[:5])
