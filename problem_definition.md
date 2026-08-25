# Mentorship and Teamwork

*Google Hash Code 2022 Online Qualification Round.*

## Background

Work is more productive and more fun in a team. By combining different skills,
people can be more creative and efficient, and — just as importantly — they can
learn from one another. In this problem you are given a set of contributors who
have mastered various skills, and a set of projects with specific skill
requirements. Contributors improve their skills by completing projects, and they
can mentor each other to take on roles they could not handle alone.

Your task is to assign contributors to project roles that match their
qualifications so that the total score of completed projects is maximized.

## Contributors

There are **C** contributors. Each contributor has:

- a **name**, and
- zero or more **skills**, each held at a non-negative integer **level**
  (0, 1, 2, …).

Not having a skill is the same as having it at level 0.

## Projects

There are **P** projects. Each project has:

- a **name**;
- a **duration** `D` — the number of days needed to finish it once it starts;
- a **score** `S` — the points awarded for completing it;
- a **best-before day** `B` — if the project's last working day is strictly
  before day `B`, it earns the full score; otherwise it loses one point for each
  day of delay, but never drops below 0 points;
- a list of **roles** that must all be filled.

Each role requires **one skill at a specific level** and is filled by exactly one
contributor. A contributor can fill at most one role in a given project.

## Filling roles and mentorship

A contributor can be assigned to a role requiring `skill` at `level` if either:

1. they have that skill at the required level or higher, **or**
2. they have it at exactly one level below the requirement, **and** another
   contributor on the same project (in a different role) has that skill at the
   required level or higher — in which case they are **mentored** by that
   colleague.

Additional notes:

- One contributor can mentor several others at once, including for the same skill.
- A contributor can mentor and be mentored at the same time.
- Since a missing skill equals level 0, a contributor with no knowledge of a
  skill can still fill a role requiring it at level 1, provided a teammate on the
  same project has that skill at level 1 or higher.

## Scheduling and assignments

- Every contributor is available starting on day 0.
- A contributor can work on at most one project at a time.
- If a contributor is assigned to several projects, they work on them in the
  order given in the output, one after another.
- A project starts on the first day on which **all** of its assigned contributors
  are available, and occupies them for `D` consecutive days. A project with
  duration `D` that starts on day `t` is worked on during days `t … t + D − 1`
  and is finished at the start of day `t + D`.

An assignment is **invalid** (and the whole submission is rejected) if, by the
time a contributor is supposed to start a role, they do not meet the skill
requirement for it (with or without mentoring), given the skill levels they have
after finishing all of their previously listed projects.

## Learning (levelling up)

Completing a project is a learning opportunity. When a project is completed, for
each contributor working on it:

- if the **required level** of their role was **equal to or higher** than their
  current level in that skill, their level in that skill increases by 1;
- otherwise their level stays the same.

Mentoring does **not** raise the mentor's skill level.

## Scoring

Each successfully completed project yields its score `S`, minus one point per day
of lateness (down to a minimum of 0). A project may score 0 and still be worked
on — its contributors can still gain skill levels from it. The total score is the
sum of the scores of all correctly completed projects.

## Input format

The input is a plain ASCII text file with Unix-style (`\n`) line endings. When a
line holds several values, they are separated by single spaces.

- **Line 1:** two integers `C P` — the number of contributors and the number of
  projects.
- Then **C contributor blocks**. Each block is:
  - a line `name N` — the contributor's name and the number `N` of their skills;
  - `N` lines, each `skill_name level`.
- Then **P project blocks**. Each block is:
  - a line `name D S B R` — the project's name, duration, score, best-before day,
    and number of roles `R`;
  - `R` lines, each `skill_name level`, listing the roles **in order**.

### Constraints (typical ranges)

- `1 ≤ C ≤ 100000`
- `1 ≤ P ≤ 100000`
- Names and skill names are ASCII strings of at most 20 characters; names use
  `a–z`, `A–Z`, `0–9`; skill names may also include `-` and `+`.
- Skill levels and required levels are non-negative integers.

## Output format

The output describes which projects you plan to carry out and who fills each role:

- **Line 1:** an integer `Q` — the number of projects you are planning.
- Then, for each of the `Q` projects:
  - a line with the **project name** (each project may appear at most once);
  - a line with the **names of the contributors** assigned to that project's
    roles, in the same order the roles appear in the input, separated by single
    spaces.

## Example

### Input

```
3 3
Anna 1
C++ 2
Bob 2
HTML 5
CSS 5
Maria 1
Python 3
Logging 5 10 5 1
C++ 3
WebServer 7 10 7 2
HTML 3
C++ 2
WebChat 10 20 20 2
Python 3
HTML 3
```

This describes 3 contributors and 3 projects:

- **Anna** — C++ 2
- **Bob** — HTML 5, CSS 5
- **Maria** — Python 3
- **Logging** — duration 5, score 10, best-before 5, 1 role: C++ ≥ 3
- **WebServer** — duration 7, score 10, best-before 7, 2 roles: HTML ≥ 3, C++ ≥ 2
- **WebChat** — duration 10, score 20, best-before 20, 2 roles: Python ≥ 3, HTML ≥ 3

### Output

```
3
WebServer
Bob Anna
Logging
Anna
WebChat
Maria Bob
```

This plans all three projects: Bob and Anna take WebServer's two roles, Anna
takes Logging's single role, and Maria and Bob take WebChat's two roles.