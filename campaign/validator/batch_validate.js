'use strict';
// Fast batch front-end for the vendored Hash Code 2022 referee.
//
// Two modes:
//   node batch_validate.js INSTANCE SOLUTION      -> one JSON line on stdout
//   node batch_validate.js --server               -> JSON-lines protocol:
//        stdin : {"id": ..., "instance": "/abs/path", "solution": "/abs/path"}
//        stdout: {"id": ..., "valid": bool, "score": n|null, "error": str|null,
//                 "strict_valid": bool, "strict_score": n|null,
//                 "strict_error": str|null, "projects": n|null, "ms": n}
//
// Speed-ups over `cp sol data/x.in.txt.out.txt && node index.js data/x.in.txt`:
//   * one long-lived Node process scores many solutions (no start-up per run);
//   * paths are read directly (no copying next to the input);
//   * recently used instances are cached in memory (text + strict parse).
//
// "valid"/"score" come from the unmodified upstream referee (submission.js).
// "strict_*" re-checks the same file with start-of-project skill levels,
// matching the official rules; see upstream/VENDORED.md for why.

const fs = require('fs');
const readline = require('readline');
const submission = require('./upstream/submission');

const CACHE_SIZE = 4;

// Upstream splits on LF and ' ' only, so a CRLF file (e.g. solver stdout on
// Windows) makes it count a lone CR as an extra contributor. Normalise first;
// on Linux the files are LF already and this is a no-op.
const CR_LF = new RegExp(String.fromCharCode(13) + String.fromCharCode(10) + '?', 'g');
const lf = s => s.replace(CR_LF, String.fromCharCode(10));
const cache = new Map(); // path -> {text, parsed}

function loadInstance(path) {
  let entry = cache.get(path);
  if (entry) {
    cache.delete(path);
    cache.set(path, entry); // LRU refresh
    return entry;
  }
  const text = lf(fs.readFileSync(path, 'utf8'));
  entry = { text, parsed: parseInstance(text) };
  cache.set(path, entry);
  while (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value);
  return entry;
}

function parseInstance(text) {
  const t = text.split(/\s+/).filter(Boolean);
  let i = 0;
  const C = +t[i++];
  const P = +t[i++];
  const contributors = new Map(); // name -> index
  const skills0 = [];             // index -> Map(skill -> level)
  for (let c = 0; c < C; c++) {
    const name = t[i++];
    const n = +t[i++];
    const m = new Map();
    for (let k = 0; k < n; k++) { m.set(t[i], +t[i + 1]); i += 2; }
    contributors.set(name, c);
    skills0.push(m);
  }
  const projects = new Map();
  for (let p = 0; p < P; p++) {
    const name = t[i++];
    const D = +t[i++], S = +t[i++], B = +t[i++], R = +t[i++];
    const roles = [];
    for (let k = 0; k < R; k++) { roles.push([t[i], +t[i + 1]]); i += 2; }
    projects.set(name, { D, S, B, roles });
  }
  return { contributors, skills0, projects };
}

// Official rules: every check uses the levels at project start; learning
// (+1 when level <= required) is applied only after the whole team is checked.
function strictScore(inst, solText) {
  const lines = solText.split('\n');
  const n = Number(lines[0] === undefined ? '' : lines[0].trim());
  if (!Number.isInteger(n) || n < 0 || lines[0].trim() === '') throw new Error('line 1: bad project count');
  if (n > inst.projects.size) throw new Error(`line 1: ${n} projects > ${inst.projects.size}`);
  const levels = new Map(); // contributor index -> Map, copy-on-write
  const lvl = (c, s) => ((levels.get(c) || inst.skills0[c]).get(s) || 0);
  const avail = new Map();
  const used = new Set();
  let score = 0;
  let ln = 1;
  for (let k = 0; k < n; k++) {
    if (lines[ln] === undefined || lines[ln + 1] === undefined) throw new Error(`line ${ln + 1}: unexpected EOF`);
    const pname = lines[ln++].trim();
    const names = lines[ln++].split(/\s+/).filter(Boolean);
    const pr = inst.projects.get(pname);
    if (!pr) throw new Error(`line ${ln - 1}: unknown project ${pname}`);
    if (used.has(pname)) throw new Error(`line ${ln - 1}: project ${pname} listed twice`);
    used.add(pname);
    if (names.length !== pr.roles.length) throw new Error(`line ${ln}: ${names.length} contributors for ${pr.roles.length} roles`);
    const team = names.map(nm => {
      const c = inst.contributors.get(nm);
      if (c === undefined) throw new Error(`line ${ln}: unknown contributor ${nm}`);
      return c;
    });
    if (new Set(team).size !== team.length) throw new Error(`line ${ln}: contributor repeated in ${pname}`);
    const pre = pr.roles.map(([s], j) => lvl(team[j], s));
    for (let j = 0; j < team.length; j++) {
      const [s, L] = pr.roles[j];
      if (pre[j] >= L) continue;
      if (pre[j] === L - 1 && team.some((c, q) => q !== j && lvl(c, s) >= L)) continue;
      throw new Error(`line ${ln}: ${names[j]} has ${s}=${pre[j]}, role needs ${L}, no mentor at project start`);
    }
    let start = 0;
    for (const c of team) start = Math.max(start, avail.get(c) || 0);
    const end = start + pr.D;
    score += Math.max(0, pr.S - Math.max(0, end - pr.B));
    for (let j = 0; j < team.length; j++) {
      const [s, L] = pr.roles[j];
      const c = team[j];
      avail.set(c, end);
      if (pre[j] <= L) {
        if (!levels.has(c)) levels.set(c, new Map(inst.skills0[c]));
        levels.get(c).set(s, pre[j] + 1);
      }
    }
  }
  for (; ln < lines.length; ln++) if (lines[ln].trim()) throw new Error(`line ${ln + 1}: trailing content`);
  return { score, projects: n };
}

function validate(instancePath, solutionPath) {
  const t0 = process.hrtime.bigint();
  const out = { valid: false, score: null, error: null, strict_valid: false, strict_score: null, strict_error: null, projects: null };
  try {
    const inst = loadInstance(instancePath);
    const solText = lf(fs.readFileSync(solutionPath, 'utf8'));
    try {
      out.score = submission({ inputDataSet: inst.text, submittedDataSet: solText, outputToFile: 1 }).evaluate().score;
      out.valid = true;
    } catch (e) { out.error = String(e && e.message || e); }
    try {
      const r = strictScore(inst.parsed, solText);
      out.strict_valid = true; out.strict_score = r.score; out.projects = r.projects;
    } catch (e) { out.strict_error = String(e && e.message || e); }
  } catch (e) {
    out.error = out.strict_error = `io: ${e && e.message || e}`;
  }
  out.ms = Number(process.hrtime.bigint() - t0) / 1e6;
  return out;
}

const args = process.argv.slice(2);
if (args[0] === '--server') {
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on('line', line => {
    if (!line.trim()) return;
    let req;
    try { req = JSON.parse(line); } catch (e) {
      process.stdout.write(JSON.stringify({ id: null, valid: false, error: 'bad request json' }) + '\n');
      return;
    }
    process.stdout.write(JSON.stringify({ id: req.id, ...validate(req.instance, req.solution) }) + '\n');
  });
} else if (args.length === 2) {
  const r = validate(args[0], args[1]);
  console.log(JSON.stringify(r));
  process.exitCode = r.valid && r.strict_valid ? 0 : 1;
} else {
  console.error('usage: node batch_validate.js INSTANCE SOLUTION | --server');
  process.exitCode = 2;
}
