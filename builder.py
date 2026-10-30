"""Assembles the manuscript inside the user's IEEE Transactions master template (styles, page set-up,
two-column body, caption styles). Header and footer are left empty (footer references removed)."""
import re, zipfile, shutil, html, os
from PIL import Image
import omml

TEMPLATE = "/mnt/user-data/attachments/9e1a125c-86c8-5379-84b9-ea15176f02fa/ebdbc7a8-Q1_SCI_IEEE_Transactions_Master_Template.docx"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]
PG = '<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1008" w:right="936" w:bottom="1080" w:left="936" w:header="720" w:footer="720" w:gutter="0"/>'
SECT_1COL = f'<w:sectPr>{PG}<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
SECT_2COL_CONT = f'<w:sectPr><w:type w:val="continuous"/>{PG}<w:cols w:num="2" w:space="288"/><w:docGrid w:linePitch="360"/></w:sectPr>'
SECT_1COL_CONT = f'<w:sectPr><w:type w:val="continuous"/>{PG}<w:cols w:space="288"/><w:docGrid w:linePitch="360"/></w:sectPr>'
COL_W, FULL_W = 5040, 10368  # twips


def esc(t):
    return html.escape(t, quote=False)


class Doc:
    def __init__(self):
        self.body = []; self.sec = 0; self.sub = 0; self.neq = 0; self.fig = 0; self.tab = 0; self.alg = 0
        self.images = []; self.cites = {}; self.cite_order = []; self.labels = {}; self.n_display = 0
        self.n_inline = 0

    # ---------------------------------------------------------------- inline markup
    def cite(self, keys):
        nums = []
        for k in keys:
            if k not in self.cites:
                self.cite_order.append(k); self.cites[k] = len(self.cite_order)
            nums.append(self.cites[k])
        nums = sorted(set(nums))
        assert len(nums) <= 3, f"more than 3 references in one bracket: {keys}"
        if len(nums) == 3 and nums[2] - nums[0] == 2:
            return f"[{nums[0]}-{nums[2]}]"
        return "[" + ", ".join(str(n) for n in nums) + "]"

    def runs(self, text, size=None, bold=False, italic=False):
        """Text with $math$, **bold**, *italic*, {cite:a,b} and {ref:label}."""
        text = re.sub(r"\{cite:([^}]+)\}", lambda m: self.cite([k.strip() for k in m.group(1).split(",")]), text)
        text = re.sub(r"\{ref:([^}]+)\}", lambda m: self.labels.get(m.group(1), "??" + m.group(1)), text)
        out = []
        parts = re.split(r"(\$[^$]+\$)", text)
        for p in parts:
            if not p:
                continue
            if p.startswith("$") and p.endswith("$"):
                out.append(omml.inline(p[1:-1], size)); self.n_inline += 1; continue
            for q in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", p):
                if not q:
                    continue
                b, i = bold, italic
                if q.startswith("**"):
                    q, b = q[2:-2], True
                elif q.startswith("*") and q.endswith("*") and len(q) > 2:
                    q, i = q[1:-1], True
                rpr = ("<w:b/>" if b else "") + ("<w:i/>" if i else "") + (f'<w:sz w:val="{size}"/>' if size else "")
                out.append(f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{esc(q)}</w:t></w:r>')
        return "".join(out)

    # ---------------------------------------------------------------- blocks
    def raw(self, x):
        self.body.append(x)

    def title(self, t, authors, affil):
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="PaperTitle"/></w:pPr><w:r><w:t xml:space="preserve">{esc(t)}</w:t></w:r></w:p>')
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="Authors"/></w:pPr><w:r><w:t xml:space="preserve">{esc(authors)}</w:t></w:r></w:p>')
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="Affiliations"/></w:pPr><w:r><w:t xml:space="preserve">{esc(affil)}</w:t></w:r></w:p>')
        self.raw("<w:p/>")

    def abstract(self, text, terms):
        self.raw('<w:p><w:pPr><w:ind w:left="360" w:right="360"/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:b/><w:i/></w:rPr>'
                 f'<w:t>Abstract—</w:t></w:r>{self.runs(text)}</w:p>')
        self.raw('<w:p><w:pPr><w:ind w:left="360" w:right="360"/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:b/><w:i/></w:rPr>'
                 f'<w:t>Index Terms—</w:t></w:r>{self.runs(terms)}</w:p>')
        self.raw(f"<w:p><w:pPr>{SECT_1COL}</w:pPr></w:p>")

    def h1(self, t, numbered=True):
        self.sub = 0
        if numbered:
            self.sec += 1; t = f"{ROMAN[self.sec - 1]}. {t}"
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="IEEESection"/><w:keepNext/></w:pPr><w:r><w:rPr><w:smallCaps/></w:rPr>'
                 f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r></w:p>')

    def h2(self, t):
        self.sub += 1
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="IEEESubsection"/><w:keepNext/></w:pPr><w:r>'
                 f'<w:t xml:space="preserve">{chr(64 + self.sub)}. {esc(t)}</w:t></w:r></w:p>')

    def h3(self, t):
        self.raw(f'<w:p><w:pPr><w:keepNext/><w:jc w:val="both"/></w:pPr><w:r><w:rPr><w:i/></w:rPr>'
                 f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r></w:p>')

    def p(self, text, keep_next=False):
        kn = "<w:keepNext/>" if keep_next else ""
        self.raw(f'<w:p><w:pPr>{kn}<w:jc w:val="both"/></w:pPr>{self.runs(text)}</w:p>')

    def bullets(self, items):
        for it in items:
            self.raw(f'<w:p><w:pPr><w:ind w:left="288" w:hanging="216"/><w:jc w:val="both"/></w:pPr>'
                     f'<w:r><w:t xml:space="preserve">•  </w:t></w:r>{self.runs(it)}</w:p>')

    def label(self, name, value):
        self.labels[name] = value

    def eq(self, latex, label=None):
        self.neq += 1; self.n_display += 1
        if label:
            self.labels[label] = f"({self.neq})"
        self.raw(omml.display(latex, self.neq))

    # ---------------------------------------------------------------- floats
    def _section_switch(self, to_full):
        self.raw(f"<w:p><w:pPr><w:spacing w:after=\"0\"/>{SECT_2COL_CONT if to_full else SECT_1COL_CONT}</w:pPr></w:p>")

    def figure(self, path, caption, label=None, full=False, width_in=None):
        self.fig += 1
        if label:
            self.labels[label] = f"Fig. {self.fig}"
        im = Image.open(path); w, h = im.size
        max_w = (FULL_W if full else COL_W) / 1440.0
        width_in = min(width_in or max_w, max_w)
        cx = int(width_in * 914400); cy = int(cx * h / w)
        rid = f"rIdImg{len(self.images) + 1}"; name = f"image{len(self.images) + 1}.png"
        self.images.append((rid, name, path))
        n = len(self.images)
        drawing = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                   f'<wp:docPr id="{n}" name="Figure {n}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
                   f'</wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                   f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{n}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>'
                   f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
                   f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
                   f'</a:graphicData></a:graphic></wp:inline></w:drawing>')
        if full:
            self._section_switch(True)
        self.raw(f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr><w:r>{drawing}</w:r></w:p>')
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="FigureCaption"/><w:spacing w:after="160"/><w:jc w:val="both"/></w:pPr>'
                 f'{self.runs(f"Fig. {self.fig}. " + caption)}</w:p>')
        if full:
            self._section_switch(False)

    def table(self, title, header, rows, widths=None, label=None, full=False, note=None, bold_rows=(), size=14,
              header_rows=1, align=None):
        """header: list of header rows (each a list of cells). Cells are markup strings."""
        self.tab += 1
        if label:
            self.labels[label] = f"Table {ROMAN[self.tab - 1]}"
        total = FULL_W if full else COL_W
        ncol = len(rows[0]) if rows else len(header[0])
        widths = widths or [1] * ncol
        tw = [int(total * w / sum(widths)) for w in widths]
        if full:
            self._section_switch(True)
        self.raw(f'<w:p><w:pPr><w:pStyle w:val="TableCaption"/><w:keepNext/><w:spacing w:before="120" w:after="60"/></w:pPr>'
                 f'<w:r><w:t>TABLE {ROMAN[self.tab - 1]}</w:t></w:r><w:r><w:br/></w:r>{self.runs(title.upper())}</w:p>')
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in tw)
        x = (f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{sum(tw)}" w:type="dxa"/><w:jc w:val="center"/>'
             f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="57" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar>'
             f'<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
             f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>')
        all_rows = [(r, True) for r in header] + [(r, False) for r in rows]
        for ri, (r, is_head) in enumerate(all_rows):
            x += '<w:tr><w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if is_head else '') + '<w:jc w:val="center"/></w:trPr>'
            ci = 0
            for cell in r:
                span = 1
                if isinstance(cell, tuple):
                    cell, span = cell
                wcell = sum(tw[ci: ci + span])
                shd = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if is_head else ""
                gs = f'<w:gridSpan w:val="{span}"/>' if span > 1 else ""
                bold = is_head or (ri - len(header)) in bold_rows
                jc = "center" if (is_head or (align and align[ci] == "c") or (align is None and ci > 0)) else "left"
                keep = "<w:keepNext/>" if ri < len(all_rows) - 1 else ""
                x += (f'<w:tc><w:tcPr><w:tcW w:w="{wcell}" w:type="dxa"/>{gs}{shd}<w:vAlign w:val="center"/></w:tcPr>'
                      f'<w:p><w:pPr>{keep}<w:spacing w:after="0"/><w:jc w:val="{jc}"/></w:pPr>{self.runs(str(cell), size=size, bold=bold)}</w:p></w:tc>')
                ci += span
            x += "</w:tr>"
        x += "</w:tbl>"
        self.raw(x)
        if note:
            self.raw(f'<w:p><w:pPr><w:spacing w:before="40" w:after="160"/><w:jc w:val="both"/></w:pPr>{self.runs(note, size=14)}</w:p>')
        else:
            self.raw('<w:p><w:pPr><w:spacing w:after="100"/></w:pPr></w:p>')
        if full:
            self._section_switch(False)

    def algorithm(self, title, lines, label=None):
        """Single-column boxed pseudocode with numbered lines (top/bottom rules)."""
        self.alg += 1
        if label:
            self.labels[label] = f"Algorithm {self.alg}"
        top = '<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        mid = '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        bot = '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        x = (f'<w:tbl><w:tblPr><w:tblW w:w="{COL_W}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
             f'<w:tblCellMar><w:left w:w="57" w:type="dxa"/><w:right w:w="57" w:type="dxa"/></w:tblCellMar></w:tblPr>'
             f'<w:tblGrid><w:gridCol w:w="{COL_W}"/></w:tblGrid>')
        x += (f'<w:tr><w:trPr><w:cantSplit/></w:trPr><w:tc><w:tcPr><w:tcW w:w="{COL_W}" w:type="dxa"/><w:tcBorders>{top}{mid}</w:tcBorders></w:tcPr>'
              f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="0"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="16"/></w:rPr>'
              f'<w:t xml:space="preserve">Algorithm {self.alg}: </w:t></w:r>{self.runs(title, size=16)}</w:p></w:tc></w:tr>')
        body = ""
        n = 0
        for ln in lines:
            indent = 0
            while ln.startswith("  "):
                indent += 1; ln = ln[2:]
            if ln.startswith("#"):
                lab = ""; ln = ln[1:].strip()
            else:
                n += 1; lab = f"{n}: "
            body += (f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="0"/><w:ind w:left="{indent * 220}"/></w:pPr>'
                     f'<w:r><w:rPr><w:sz w:val="15"/></w:rPr><w:t xml:space="preserve">{lab}</w:t></w:r>{self.runs(ln, size=15)}</w:p>')
        x += (f'<w:tr><w:trPr><w:cantSplit/></w:trPr><w:tc><w:tcPr><w:tcW w:w="{COL_W}" w:type="dxa"/><w:tcBorders>{bot}</w:tcBorders></w:tcPr>'
              f'{body}</w:tc></w:tr></w:tbl>')
        self.raw('<w:p><w:pPr><w:spacing w:after="0"/></w:pPr></w:p>')
        self.raw(x)
        self.raw('<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>')

    def references(self, refs):
        """refs: dict key -> formatted IEEE string (markup). Emitted in citation order."""
        missing = [k for k in refs if k not in self.cites]
        assert not missing, f"uncited references: {missing}"
        unknown = [k for k in self.cite_order if k not in refs]
        assert not unknown, f"cited but not in bibliography: {unknown}"
        for k in self.cite_order:
            self.raw(f'<w:p><w:pPr><w:spacing w:after="0"/><w:ind w:left="400" w:hanging="400"/><w:jc w:val="both"/></w:pPr>'
                     f'<w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:t xml:space="preserve">[{self.cites[k]}]\t</w:t></w:r>'
                     f'{self.runs(refs[k], size=16)}</w:p>')

    # ---------------------------------------------------------------- save
    def save(self, out):
        z = zipfile.ZipFile(TEMPLATE)
        files = {n: z.read(n) for n in z.namelist()}
        doc = files["word/document.xml"].decode("utf8")
        head = doc[: doc.find("<w:body>") + len("<w:body>")]
        final = f'<w:sectPr><w:type w:val="continuous"/>{PG}<w:cols w:num="2" w:space="288"/><w:docGrid w:linePitch="360"/></w:sectPr>'
        body = "".join(self.body)
        # the first block (front matter) ends with a 1-col sectPr; the body is 2-col
        newdoc = head + body + final + "</w:body></w:document>"
        for ns, uri in (("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
                        ("a", "http://schemas.openxmlformats.org/drawingml/2006/main"),
                        ("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture")):
            if f"xmlns:{ns}=" not in newdoc[:3000]:
                newdoc = newdoc.replace("<w:document ", f'<w:document xmlns:{ns}="{uri}" ', 1)
        files["word/document.xml"] = newdoc.encode("utf8")
        rels = files["word/_rels/document.xml.rels"].decode("utf8")
        rels = re.sub(r'<Relationship [^>]*Type="[^"]*/footer"[^>]*/>', "", rels)
        for rid, name, _ in self.images:
            rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
        files["word/_rels/document.xml.rels"] = rels.encode("utf8")
        ct = files["[Content_Types].xml"].decode("utf8")
        ct = re.sub(r'<Override PartName="/word/footer\d\.xml"[^>]*/>', "", ct)
        if 'Extension="png"' not in ct:
            ct = ct.replace("<Types ", "<Types ", 1).replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
        files["[Content_Types].xml"] = ct.encode("utf8")
        for n in list(files):
            if re.match(r"word/footer\d\.xml", n):
                del files[n]
        for rid, name, path in self.images:
            files[f"word/media/{name}"] = open(path, "rb").read()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as o:
            for n, b in files.items():
                o.writestr(n, b)
        return out
