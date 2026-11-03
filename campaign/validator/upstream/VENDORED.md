# Vendored validator (unmodified)

Source: https://github.com/KristofHolecz/hash-code-2022-mentorship-and-teamwork-insights-reporter
Commit: `f75d6e29c278d1146b1e5ec23e7fcdfce4e1d685` (2022-05-05)
License: ISC (declared in `package.json`; author Kristóf Holecz)

Files copied verbatim: `index.js`, `submission.js`, `utils.js`, `package.json`,
`README.md`. The upstream `data/` folder (the Google inputs) and `docs/` PDF were
not copied.

SHA-256:

```
fd7ca911bde971013992352028fba391add4a4fc664ea9c7752ce9910ad68de3  index.js
5c17fb5e3ef7ff79da879b18c3614acf86898e580c2ad849fdadd03f52b68d84  submission.js
267602641d6159f26c2061ac0cce4113a937735edf3675d356e205db98bbb462  utils.js
```

The campaign calls `submission.js` directly from `../batch_validate.js`, so
no copy to `data/<input>.out.txt` and no per-run `node index.js` is needed.
`node index.js data/<input>` still works exactly as upstream documents.

## Known leniency of upstream

The upstream referee checks the mentor condition against skill levels that
have *already been incremented by earlier roles of the same project*. Example:
A and B both have X=3, the project needs roles (X 3, X 4). Under the official
rules nobody has X>=4 when the project starts, so the assignment is invalid. The
upstream referee raises A to 4 while processing role 1 and then accepts A as
B's mentor for role 2 (score 10). `batch_validate.js` therefore also runs a
strict check that uses start-of-project levels, and reports any disagreement.
