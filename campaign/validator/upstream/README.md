# hash-code-2022-mentorship-and-teamwork-insights-reporter

Referee and insights reporter for [Google Hash Code 2022 Qualification Round (Mentorship and Teamwork)](https://codingcompetitions.withgoogle.com/hashcode/archive) with zero dependencies. It returns the same result as the Insights pane in the Submissions and score menu in the Google Hash Code Judge System.

## Usage

The input data set and the submission file(s) need to be placed in the same folder with the following naming convention:

Input data set: `a_an_example.in.txt` \
Submission: `a_an_example.in.txt.out.txt`

Evaluate the submission and print the insights to stdout:

```bash
$ node index.js data/a_an_example.in.txt
```

Evaluate the submission and write the insights to file (it creates a file called `a_an_example.in.txt.insights.txt` and will not print anything to stdout):

```bash
$ node index.js data/a_an_example.in.txt 1
```

Having trouble using the referee and insights reporter, just drop the following line into the CLI.

```bash
$ node index.js
```

Note that the referee is not prepared for all cases of incorrect submission file.
