"""LaTeX -> native Word OMML (editable equations), following the template's conventions:
Cambria Math run properties on every m:r, and numbered display equations as a 4-column
borderless matrix [blank | equation | blank | (n)] inside m:oMathPara."""
import re, subprocess, tempfile, os, zipfile, html

RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:lang w:val="en-IN"/></w:rPr>'
CTRL = f'<m:ctrlPr>{RPR}</m:ctrlPr>'
_cache = {}


def _fix(x):
    x = x.replace(" />", "/>")
    # schema order inside m:dPr: begChr, sepChr, endChr, grow, shp, ctrlPr
    def dpr(m):
        body = m.group(1)
        parts = re.findall(r"<m:(begChr|sepChr|endChr|grow|shp)[^>]*/>", body)
        items = {k: re.search(rf"<m:{k}[^>]*/>", body).group(0) for k in parts}
        order = [items[k] for k in ("begChr", "sepChr", "endChr", "grow", "shp") if k in items]
        return "<m:dPr>" + "".join(order) + CTRL + "</m:dPr>"
    x = re.sub(r"<m:dPr>(.*?)</m:dPr>", dpr, x)
    # m:rPr order: lit, nor | (scr, sty), brk, aln
    def rpr(m):
        body = m.group(1)
        keys = ["lit", "nor", "scr", "sty", "brk", "aln"]
        found = {k: re.search(rf"<m:{k}[^>]*/>", body) for k in keys}
        return "<m:rPr>" + "".join(found[k].group(0) for k in keys if found[k]) + "</m:rPr>"
    x = re.sub(r"<m:rPr>(.*?)</m:rPr>", rpr, x)
    # Cambria Math run properties after m:rPr (or at the start of m:r)
    x = re.sub(r"<m:r>(<m:rPr>.*?</m:rPr>)?", lambda m: "<m:r>" + (m.group(1) or "") + RPR, x)
    # control properties for structures without them
    for tag in ("sSubSupPr", "sSubPr", "sSupPr", "fPr", "radPr", "naryPr", "accPr", "barPr", "funcPr", "limLowPr",
                "limUppPr", "groupChrPr", "eqArrPr", "mPr", "boxPr", "borderBoxPr", "sPrePr"):
        x = re.sub(rf"<m:{tag}>(.*?)</m:{tag}>", lambda m: f"<m:{tag}>{m.group(1)}{CTRL}</m:{tag}>" if "ctrlPr" not in m.group(1) else m.group(0), x)
        x = x.replace(f"<m:{tag}/>", f"<m:{tag}>{CTRL}</m:{tag}>")
    # structures emitted without a property element
    for tag, pr in (("sSub", "sSubPr"), ("sSup", "sSupPr"), ("sSubSup", "sSubSupPr"), ("f", "fPr"), ("rad", "radPr"),
                    ("func", "funcPr"), ("acc", "accPr"), ("bar", "barPr"), ("limLow", "limLowPr"), ("limUpp", "limUppPr")):
        x = re.sub(rf"<m:{tag}>(?!<m:{pr})", f"<m:{tag}><m:{pr}>{CTRL}</m:{pr}>", x)
    return x


def convert(snippets):
    """Convert a list of LaTeX strings to OMML inner content (children of m:oMath)."""
    todo = [s for s in dict.fromkeys(snippets) if s not in _cache]
    if todo:
        with tempfile.TemporaryDirectory() as d:
            md = "\n\n".join(f"P{i}: ${s}$" for i, s in enumerate(todo))
            open(f"{d}/a.md", "w").write(md)
            subprocess.run(["pandoc", f"{d}/a.md", "-f", "markdown", "-o", f"{d}/a.docx"], check=True)
            xml = zipfile.ZipFile(f"{d}/a.docx").read("word/document.xml").decode("utf8")
        paras = re.findall(r"<w:p>.*?</w:p>|<w:p .*?</w:p>", xml, re.S)
        got = {}
        for p in paras:
            m = re.search(r"P(\d+):", "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)))
            om = re.search(r"<m:oMath>(.*)</m:oMath>", p, re.S)
            if m and om:
                got[int(m.group(1))] = _fix(om.group(1))
        for i, s in enumerate(todo):
            if i not in got:
                raise ValueError(f"pandoc failed on: {s}")
            _cache[s] = got[i]
    return [_cache[s] for s in snippets]


def inline(latex, size=None):
    x = convert([latex])[0]
    if size:
        x = x.replace('<w:lang w:val="en-IN"/></w:rPr>', f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:lang w:val="en-IN"/></w:rPr>')
    return f"<m:oMath>{x}</m:oMath>"


def display(latex, n):
    body = convert([latex])[0]
    mpr = (f'<m:mPr><m:plcHide m:val="1"/><m:mcs><m:mc><m:mcPr><m:count m:val="4"/><m:mcJc m:val="center"/>'
           f'</m:mcPr></m:mc></m:mcs>{CTRL}</m:mPr>')
    num = f'<m:r><m:rPr><m:nor/></m:rPr><w:rPr><w:lang w:val="en-IN"/></w:rPr><m:t>({n})</m:t></m:r>'
    brk = f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr>{RPR}<w:br/></m:r>'
    return (f'<w:p><w:pPr><w:jc w:val="both"/><w:rPr><w:lang w:val="en-IN"/></w:rPr></w:pPr><m:oMathPara><m:oMath>'
            f'<m:m>{mpr}<m:mr><m:e/><m:e>{body}</m:e><m:e/><m:e>{num}</m:e></m:mr></m:m>{brk}</m:oMath></m:oMathPara></w:p>')
