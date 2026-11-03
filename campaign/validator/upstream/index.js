'use strict';

const submission = require('./submission');
const {
  readFile, writeFile, getMemoryUsageMessage, elapsedTime,
} = require('./utils');

const [filePath, outputToFile = 0, debug = 0] = process.argv.slice(2);

if (!filePath || isNaN(outputToFile) || isNaN(debug)) {
  console.info(`Usage:
  node index.js <input-file-path> [output-to-file] [debug]

Options:
  input-file-path  The path of the input data set.
  output-to-file   Write the insights to stdout (0) or file (1) [default: 0].
  debug            Write memory usage and duration to the stdout [default: 0].

Examples:
  node index.js data/a.txt
    This will parse the given input data set (data/a.txt)
    with its submission (data/a.txt.out.txt) then evaluate and
    print the insights to the stdout.

  node index.js data/d.txt 1
    This will parse the given input data set (data/d.txt)
    with its submission (data/d.txt.out.txt) then evaluate and
    create a file (data/d.txt.insights.txt) containing insights.
`);
  return;
}

const run = async () => {
  const elapsed = elapsedTime(+debug);

  try {
    const [inputDataSet, submittedDataSet] = await Promise.all([
      readFile(filePath), readFile(`${filePath}.out.txt`),
    ]);
    const output = submission({ inputDataSet, submittedDataSet, outputToFile })
      .evaluate()
      .toString();

    if (!+outputToFile) {
      console.info(output);
      return;
    }

    await writeFile(`${filePath}.insights.txt`, output);
  } catch (error) {
    console.error(error.message);
  } finally {
    elapsed.log(`${getMemoryUsageMessage()}\nDuration`).reset();
  }
};

run();
