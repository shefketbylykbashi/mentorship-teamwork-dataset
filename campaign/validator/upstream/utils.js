'use strict';

const fs = require('fs');
const { join } = require('path');

const readFile = (path, encoding = 'utf8') => new Promise((resolve, reject) => {
  fs.readFile(join(__dirname, path), encoding, (error, result) =>
    error ? reject(error) : resolve(result)
  );
});

const writeFile = (path, data) => new Promise((resolve, reject) => {
  fs.writeFile(join(__dirname, path), data, (error, result) =>
    error ? reject(error) : resolve(result)
  );
});

const getMemoryUsageMessage = () => {
  const labelify = (value, convertToUpperCase) => {
    const result = value.charAt(0).toUpperCase() +
      value.substr(1).replace(/([a-z0-9])([A-Z])/g, '$1 $2');

    return convertToUpperCase ? result.toUpperCase() : result;
  };
  const memoryUsage = process.memoryUsage();
  const message = Object.keys(memoryUsage)
    .map((key, i) =>
      `${labelify(key, !i)}: ${
        (memoryUsage[key] / 1024 / 1024).toFixed(2)
      }`
    )
    .join(' :: ');

  return `Memory Usage (MiB) [${message}]`;
};

const elapsedTime = (debug = 1) => {
  let start = process.hrtime.bigint();

  const timer = {
    log: message => {
      if (debug) {
        const diff = process.hrtime.bigint() - start;

        console.debug([
          message && `${message}:`,
          `${(diff / 1000000n).toLocaleString('en-US')}.${diff % 1000000n} ms`,
        ].join(' '));
      }

      return timer;
    },
    reset: () => {
      start = process.hrtime.bigint();
      return timer;
    },
  };

  return timer;
};

const parseLine = (rawLine, stringToNumberArrayIndexes = []) => {
  const parsedLine = rawLine.split(' ').filter(value => value.length);

  return stringToNumberArrayIndexes.length
    ? parsedLine.map((value, i) =>
      stringToNumberArrayIndexes.includes(i) ? +value : value
    )
    : parsedLine;
};

module.exports = {
  readFile,
  writeFile,
  getMemoryUsageMessage,
  elapsedTime,
  parseLine,
};
