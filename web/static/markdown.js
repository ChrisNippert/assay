// Markdown with math and highlighted code, shared by the chat and the
// benchmark answers. Small and safe: everything is escaped first. Math needs
// KaTeX and highlighting hljs on the page; without them it falls back to
// plain text.
window.assayMarkdown = (() => {
  // ---- markdown (small and safe: everything is escaped first) ----
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // Inline marks. Code spans first (their insides stay as typed), then
  // links, then emphasis; everything is escaped before any tag is added.
  // Links only go to http(s) and mailto addresses.
  function inline(text) {
    return text
      .split('`')
      .map((part, i) => (i % 2 ? `<code>${esc(part)}</code>` : marks(part)))
      .join('');
  }

  const SAFE_URL = /^(https?:\/\/|mailto:)/i;
  function marks(raw) {
    const links = [];
    const keep = (html) => { links.push(html); return `\u0001${links.length - 1}\u0001`; };
    let t = raw
      .replace(/\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (m, label, url) =>
        SAFE_URL.test(url) ? keep(`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${emphasis(esc(label))}</a>`) : m)
      .replace(/<(https?:\/\/[^\s>]+)>/g, (m, url) => keep(link(url)))
      .replace(/\bhttps?:\/\/[^\s<>()]+[^\s<>().,;:!?'"*_]/g, (url) => keep(link(url)));
    t = emphasis(esc(t));
    return t.replace(/\u0001(\d+)\u0001/g, (_, i) => links[i]);
  }
  const link = (url) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>`;

  function emphasis(t) {
    return t
      .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, '$1<strong>$2</strong>')
      .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>')
      .replace(/(^|[^*\w])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>')
      .replace(/(^|[^\w])_(?=[^\s_])([^_\n]*?[^\s_])_(?!\w)/g, '$1<em>$2</em>');
  }

  // ---- tables (GitHub style): a header row, a |---|:--:| row, then rows ----
  const TABLE_SEP = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;
  // Splits a row on | outside code spans and \| escapes.
  function cells(line) {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
    const out = [];
    let cur = '';
    let code = false;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (c === '\\' && s[i + 1] === '|') { cur += '|'; i++; continue; }
      if (c === '`') code = !code;
      if (c === '|' && !code) { out.push(cur.trim()); cur = ''; continue; }
      cur += c;
    }
    out.push(cur.trim());
    return out;
  }
  function table(head, sep, rows) {
    const align = cells(sep).map((c) => (c.startsWith(':') && c.endsWith(':') ? 'center' : c.endsWith(':') ? 'right' : ''));
    const cols = Math.max(head.length, align.length);
    const cell = (tag, text, i) => `<${tag}${align[i] ? ` style="text-align:${align[i]}"` : ''}>${inline(text ?? '')}</${tag}>`;
    const row = (tag, cs) => `<tr>${Array.from({ length: cols }, (_, i) => cell(tag, cs[i], i)).join('')}</tr>`;
    return `<div class="table-wrap"><table><thead>${row('th', head)}</thead>` +
      `<tbody>${rows.map((r) => row('td', cells(r))).join('')}</tbody></table></div>`;
  }

  // Block structure: headings, paragraphs, nested lists (task lists too),
  // quotes, rules and tables.
  function blocks(text) {
    const lines = text.split('\n');
    let html = '';
    let para = [];
    let stack = []; // open lists: { tag, indent, items: [html], open: bool }
    const flushPara = () => {
      if (para.length) html += `<p>${para.map(inline).join('<br>')}</p>`;
      para = [];
    };
    const closeList = (to = 0) => {
      while (stack.length > to) {
        const l = stack.pop();
        const out = `<${l.tag}>${l.items.map((i) => `<li${i.task ? ' class="task"' : ''}>${i.html}</li>`).join('')}</${l.tag}>`;
        if (stack.length) stack[stack.length - 1].items.at(-1).html += out;
        else html += out;
      }
    };
    const item = (indent, tag, body) => {
      flushPara();
      while (stack.length && indent < stack.at(-1).indent) closeList(stack.length - 1);
      const top = stack.at(-1);
      if (!top || indent > top.indent) stack.push({ tag, indent, items: [] });
      else if (top.tag !== tag) { closeList(stack.length - 1); stack.push({ tag, indent, items: [] }); }
      const task = body.match(/^\[([ xX])\]\s+(.*)$/);
      stack.at(-1).items.push(task
        ? { task: true, html: `<input type="checkbox" disabled${task[1] !== ' ' ? ' checked' : ''}> ${inline(task[2])}` }
        : { html: inline(body) });
    };
    for (let n = 0; n < lines.length; n++) {
      const line = lines[n];
      let m;
      if (!line.trim()) { flushPara(); closeList(); continue; }
      if ((m = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/)) && !/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        item(m[1].replace(/\t/g, '    ').length, /\d/.test(m[2]) ? 'ol' : 'ul', m[3]);
        continue;
      }
      if (stack.length && /^\s{2,}\S/.test(line)) { // a list item's next line
        stack.at(-1).items.at(-1).html += '<br>' + inline(line.trim());
        continue;
      }
      closeList();
      if ((m = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/))) {
        flushPara();
        const level = Math.min(m[1].length + 2, 6);
        html += `<h${level}>${inline(m[2])}</h${level}>`;
      } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        flushPara();
        html += '<hr>';
      } else if (/^\s*>/.test(line)) {
        flushPara();
        const quote = [];
        while (n < lines.length && /^\s*>/.test(lines[n])) quote.push(lines[n++].replace(/^\s*>\s?/, ''));
        n--;
        html += `<blockquote>${blocks(quote.join('\n'))}</blockquote>`;
      } else if (line.includes('|') && n + 1 < lines.length && TABLE_SEP.test(lines[n + 1]) && lines[n + 1].includes('-')) {
        flushPara();
        const head = cells(line);
        const sep = lines[n + 1];
        const rows = [];
        n += 2;
        while (n < lines.length && lines[n].includes('|') && lines[n].trim()) rows.push(lines[n++]);
        n--;
        html += table(head, sep, rows);
      } else {
        para.push(line);
      }
    }
    flushPara();
    closeList();
    return html;
  }

  // Extensions for Download, by fence label (lowercased).
  const EXT = { python: 'py', py: 'py', javascript: 'js', js: 'js', typescript: 'ts', ts: 'ts', go: 'go', rust: 'rs', rs: 'rs', c: 'c', cpp: 'cpp', 'c++': 'cpp', java: 'java', kotlin: 'kt', ruby: 'rb', rb: 'rb', php: 'php', bash: 'sh', sh: 'sh', shell: 'sh', zsh: 'sh', json: 'json', yaml: 'yaml', yml: 'yaml', toml: 'toml', html: 'html', css: 'css', sql: 'sql', markdown: 'md', md: 'md', xml: 'xml', lua: 'lua', swift: 'swift', csharp: 'cs', cs: 'cs', latex: 'tex', tex: 'tex' };
  const WHOLE_NAME = { dockerfile: 'Dockerfile', makefile: 'Makefile' };

  // A file named in a comment on the first line (// main.go), else snippet.<ext>.
  function fileNameFor(code, lang) {
    const first = code.split('\n', 1)[0];
    const m = first.match(/^\s*(?:\/\/|#|--|\/\*|<!--)\s*([\w./-]+\.\w+)\s*(?:\*\/|-->)?\s*$/);
    if (m) return m[1].split('/').pop();
    const l = lang.toLowerCase();
    return WHOLE_NAME[l] || `snippet.${EXT[l] || 'txt'}`;
  }

  function codeBlock(code, lang) {
    return `<div class="code"><div class="code-head"><span>${esc(lang)}</span>` +
      '<span class="code-actions"><button type="button" class="copy-btn" title="Copy code">Copy</button>' +
      '<button type="button" class="copy-btn dl-btn" title="Save as a file">Download</button></span></div>' +
      `<pre>${codeInner(code, lang)}</pre></div>`;
  }

  // Highlights only when the fence names a language hljs knows; no
  // auto-detect (slow, and wrong on short or half-streamed snippets).
  function codeInner(code, lang) {
    const l = (lang || '').trim().toLowerCase().split(/\s+/)[0];
    if (window.hljs && l && /^[\w+#.-]+$/.test(l) && hljs.getLanguage(l)) {
      try {
        const v = hljs.highlight(code, { language: l, ignoreIllegals: true }).value;
        return `<code class="hljs language-${l}">${v}</code>`;
      } catch { /* fall through to plain */ }
    }
    return `<code>${esc(code)}</code>`;
  }

  // Math: $$…$$ and \[…\] as blocks, $…$ and \(…\) inline. Inline code
  // spans are matched first so `$x` in code stays as typed. A single $ needs
  // no space inside it and no digit after it, so "$5 and $10" stays text.
  const MATH = /(`[^`\n]*`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$(?!\s)([^$\n`]+?)(?<!\s)\$(?!\d)/g;

  // Swaps math for placeholders the markdown pass leaves alone, renders the
  // prose, then puts KaTeX output in their place.
  function proseWithMath(text) {
    if (!window.katex) return blocks(text);
    const parts = [];
    const masked = text.replace(MATH, (m, code, d1, d2, i1, i2) => {
      if (code) return m;
      const display = d1 != null || d2 != null;
      const tex = d1 ?? d2 ?? i1 ?? i2;
      let html;
      try { html = katex.renderToString(tex, { displayMode: display, throwOnError: false }); }
      catch { return m; }
      parts.push(html);
      return `\u0000${parts.length - 1}\u0000`;
    });
    return blocks(masked).replace(/\u0000(\d+)\u0000/g, (_, i) => parts[i]);
  }

  // Splits out ``` fences; the rest is prose. An unclosed fence (mid-stream)
  // still renders as code.
  function renderMarkdown(src) {
    let html = '';
    let prose = [];
    let code = null;
    let lang = '';
    const flushProse = () => {
      if (prose.length) html += proseWithMath(prose.join('\n'));
      prose = [];
    };
    for (const line of src.split('\n')) {
      const fence = line.match(/^\s*```\s*([\w+#.-]*)/);
      if (fence && code === null) {
        flushProse();
        code = [];
        lang = fence[1];
      } else if (fence) {
        html += codeBlock(code.join('\n'), lang);
        code = null;
      } else if (code !== null) {
        code.push(line);
      } else {
        prose.push(line);
      }
    }
    if (code !== null) html += codeBlock(code.join('\n'), lang);
    flushProse();
    return html;
  }

  return { render: renderMarkdown, fileNameFor };
})();
